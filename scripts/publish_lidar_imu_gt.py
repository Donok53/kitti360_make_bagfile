#!/usr/bin/env python3
import rospy
import os
import subprocess
import numpy as np
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Imu, PointField
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import quaternion_from_euler, quaternion_from_matrix

# --- 센서 스펙 ---
# HDL-64E 실제 수직 각도 (degrees)
VEL_64_VERTICAL_ANGLES = np.array([
   -24.33, -23.07, -21.8,  -20.53, -19.26, -18.0,  -16.73, -15.46,
   -14.2,  -12.93, -11.66, -10.39,  -9.12,  -7.86,  -6.59,  -5.32,
    -4.05,  -2.79,  -1.52,  -0.25,   1.02,   2.29,   3.56,   4.83,
     6.10,   7.37,   8.64,   9.91,  11.18,  12.46,  13.73,  15.00,
    16.27,  17.54,  18.81,  20.08,  21.35,  22.62,  23.89,  25.17,
    26.44,  27.71,  28.99,  30.26,  31.53,  32.81,  34.08,  35.36,
    36.63,  37.91,  39.18,  40.46,  41.73,  43.01,  44.28,  45.56,
    46.83,  48.11,  49.38,  50.66,  51.93,  53.21,  54.48,  55.76
], dtype=np.float32)

N_SCAN = 64
DEFAULT_SCAN_PERIOD = 0.1  # 10Hz로 퍼블리시

def read_bin(filepath):
    pts4 = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    # 유효한 포인트만 필터링
    valid_mask = np.isfinite(pts4).all(axis=1)
    pts4 = pts4[valid_mask]
    
    # 비정상적으로 큰 값 제거 (센서 노이즈 등)
    distance_mask = np.linalg.norm(pts4[:, :3], axis=1) < 100  # 100m 이상 제거
    pts4 = pts4[distance_mask]
    
    return pts4

def parse_imu_line(line):
    f = list(map(float, line.strip().split()))
    # KITTI-360 IMU 데이터 형식에 맞게 수정
    # f[3], f[4], f[5]: roll, pitch, yaw (rad)
    # f[15], f[16], f[17]: af, al, au (가속도)
    # f[18], f[19], f[20]: wx, wy, wz (각속도)
    
    # 쿼터니언 생성
    quat = quaternion_from_euler(f[3], f[4], f[5])
    
    # 옵션 1: 중력 보정 없이 좌표 재배열만 (추천)
    return quat, (f[15], f[17], f[16]), (f[18], f[20], f[19])
    
    # 옵션 2: 중력 보정 포함 (기존)
    # gravity = 9.81
    # return quat, (f[15], f[17], f[16] - gravity), (f[18], f[20], f[19])
    
    # 옵션 3: X축 뒤집기 (필요시)
    # return quat, (-f[15], f[17], f[16]), (-f[18], f[20], f[19])
    
    # 옵션 4: Y축 뒤집기 (필요시)
    # return quat, (f[15], -f[17], f[16]), (f[18], -f[20], f[19])
    
    # 옵션 5: Z축 뒤집기 (필요시)
    # return quat, (f[15], f[17], -f[16]), (f[18], f[20], -f[19])

def load_gt_poses(path, T_c2l):
    poses=[]
    for L in open(path):
        v = list(map(float, L.split()))[:12]
        T = np.vstack([np.array(v).reshape(3,4), [0,0,0,1]])
        poses.append(T @ T_c2l)
    return poses

def parse_timestamp_str(s):
    dt = datetime.strptime(s[:26], "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def publish_all(lidar_dir, imu_dir, ts_path, gt_path, T_c2l, use_gt_imu=False):
    rospy.init_node('velodyne64_publisher')
    pub_pc  = rospy.Publisher("/velodyne/points", PointCloud2, queue_size=10)
    pub_imu = rospy.Publisher("/imu/data",      Imu,        queue_size=10)
    pub_gt  = rospy.Publisher("/gt_pose",       PoseStamped,queue_size=10)

    stamps = [parse_timestamp_str(l) for l in open(ts_path)]
    lfiles = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.bin'))
    ifiles = sorted(f for f in os.listdir(imu_dir)   if f.endswith('.txt')) if not use_gt_imu else []
    gtposes= load_gt_poses(gt_path, T_c2l)

    L = min(len(stamps), len(lfiles), len(gtposes))
    if not use_gt_imu:
        L = min(L, len(ifiles))
    
    stamps, lfiles, gtposes = stamps[:L], lfiles[:L], gtposes[:L]
    if not use_gt_imu:
        ifiles = ifiles[:L]

    rate = rospy.Rate(1.0/DEFAULT_SCAN_PERIOD)
    prev_stamp = None

    for i in range(L):
        t_sec = stamps[i]
        stamp = rospy.Time.from_sec(t_sec)

        # 타임스탬프 간격 체크 (400프레임 근처)
        if 390 <= i <= 410 and i > 0:
            dt_actual = t_sec - stamps[i-1]
            if dt_actual > 0.5:  # 0.5초 이상 간격이면 경고
                rospy.logwarn(f"Frame {i}: Large time gap detected: {dt_actual:.3f}s")

        # --- PointCloud2 생성 ---
        pts4 = read_bin(os.path.join(lidar_dir, lfiles[i]))
        xyz   = pts4[:,:3]
        inten = pts4[:,3]

        # 데이터 품질 체크 (400프레임 근처)
        if 390 <= i <= 410:
            rospy.logwarn(f"Frame {i}: Points={len(xyz)}, Valid points={np.sum(np.isfinite(xyz).all(axis=1))}")
            if len(xyz) < 1000:  # 포인트 수가 너무 적으면 경고
                rospy.logerr(f"Frame {i}: Too few points ({len(xyz)})")

        # 300프레임 근처 추가 디버깅
        if 290 <= i <= 310:
            rospy.logwarn(f"Frame {i}: Points={len(xyz)}, Valid points={np.sum(np.isfinite(xyz).all(axis=1))}")
            if len(xyz) < 1000:
                rospy.logerr(f"Frame {i}: Too few points ({len(xyz)})")
            
            # IMU 데이터도 체크
            if use_gt_imu:
                quat, (ax,ay,az), (gx,gy,gz) = gt_pose_to_imu(gtposes, stamps, i)
            else:
                quat, (ax,ay,az), (gx,gy,gz) = parse_imu_line(open(os.path.join(imu_dir, ifiles[i])).readline())
            
            acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
            rospy.logwarn(f"IMU[{i}]: acc_mag={acc_magnitude:.3f}, gyro_mag={gyro_magnitude:.3f}")
            
            if acc_magnitude > 50 or gyro_magnitude > 10:
                rospy.logerr(f"IMU[{i}]: Abnormal values detected!")

        # 포인트 수가 너무 적으면 이전 프레임 데이터 재사용 (간단한 보간)
        if len(xyz) < 50000:  # 5만개 미만이면 문제
            rospy.logwarn(f"Frame {i}: Low point count ({len(xyz)}), using previous frame data")
            if i > 0:
                # 이전 프레임 데이터 재사용 (간단한 보간)
                prev_pts4 = read_bin(os.path.join(lidar_dir, lfiles[i-1]))
                if len(prev_pts4) > len(xyz):
                    xyz = prev_pts4[:,:3]
                    inten = prev_pts4[:,3]
                    rospy.loginfo(f"Frame {i}: Using previous frame data ({len(xyz)} points)")

        # 추가 데이터 품질 체크
        if len(xyz) == 0:
            rospy.logerr(f"Frame {i}: No valid points, skipping frame")
            continue  # 이 프레임 스킵

        # 수직각 → ring index
        elev = np.degrees(np.arctan2(xyz[:,2], np.linalg.norm(xyz[:,:2],axis=1)))
        # 각도차 절댓값 최소 채널 선택
        ring = np.argmin(np.abs(elev[:,None] - VEL_64_VERTICAL_ANGLES[None,:]), axis=1).astype(np.uint16)

        # time field (선형보간)
        if prev_stamp is None: dt=DEFAULT_SCAN_PERIOD
        else:               dt = t_sec - prev_stamp
        times = np.linspace(0.0, dt, len(xyz), dtype=np.float32)

        header = Header(stamp=stamp, frame_id="base_link")  # LIO-SAM과 일치하도록 수정
        fields = [
            PointField('x',         0,  PointField.FLOAT32, 1),
            PointField('y',         4,  PointField.FLOAT32, 1),
            PointField('z',         8,  PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring',      16, PointField.UINT16,  1),
            PointField('time',      18, PointField.FLOAT32, 1),
        ]
        # 파이썬 기본형으로 하나씩 변환해서 리스트 컴프리헨션으로 생성
        arr = [
            (
                float(x),        # x
                float(y),        # y
                float(z),        # z
                float(intensity),# intensity
                int(r),          # ring (must be python int)
                float(t)         # time
            )
            for (x,y,z), intensity, r, t
            in zip(xyz, inten, ring, times)
        ]
        pc2_msg = pc2.create_cloud(header, fields, arr)
        pc2_msg.is_dense = True
        pub_pc.publish(pc2_msg)

        # --- IMU ---
        if use_gt_imu:
            quat, (ax,ay,az), (gx,gy,gz) = gt_pose_to_imu(gtposes, stamps, i)
        else:
            quat, (ax,ay,az), (gx,gy,gz) = parse_imu_line(open(os.path.join(imu_dir, ifiles[i])).readline())
        
        # 디버깅: 처음 몇 프레임의 IMU 데이터 출력
        if i < 5:
            rospy.loginfo(f"IMU[{i}]: acc=({ax:.3f}, {ay:.3f}, {az:.3f}), gyro=({gx:.3f}, {gy:.3f}, {gz:.3f})")
        
        # 400프레임 근처 IMU 데이터 품질 체크
        if 390 <= i <= 410:
            acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
            rospy.logwarn(f"IMU[{i}]: acc_mag={acc_magnitude:.3f}, gyro_mag={gyro_magnitude:.3f}")
            
            # 비정상적인 값 체크
            if acc_magnitude > 50 or gyro_magnitude > 10:
                rospy.logerr(f"IMU[{i}]: Abnormal values detected!")
        
        # IMU 데이터 품질 모니터링 (필터링 제거)
        if i % 100 == 0:  # 100프레임마다 한 번씩만 체크
            acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
            if acc_magnitude < 5 or acc_magnitude > 15:  # 중력가속도 범위 벗어남
                rospy.logwarn(f"IMU[{i}]: Unusual acceleration magnitude: {acc_magnitude:.3f}")
        
        imu_msg = Imu(header=Header(stamp=stamp, frame_id="base_link"))  # LIO-SAM과 일치하도록 수정
        imu_msg.orientation.x = quat[0]
        imu_msg.orientation.y = quat[1]
        imu_msg.orientation.z = quat[2]
        imu_msg.orientation.w = quat[3]
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        imu_msg.angular_velocity.x = gx  # 각속도 추가
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        pub_imu.publish(imu_msg)

        # --- GT Pose ---
        T = gtposes[i]
        q = quaternion_from_matrix(T)
        gt = PoseStamped(header=Header(stamp=stamp, frame_id="map"))
        gt.pose.position.x = T[0,3]
        gt.pose.position.y = T[1,3]
        gt.pose.position.z = T[2,3]
        gt.pose.orientation.x = q[0]
        gt.pose.orientation.y = q[1]
        gt.pose.orientation.z = q[2]
        gt.pose.orientation.w = q[3]
        pub_gt.publish(gt)

        rospy.loginfo(f"[{i+1}/{L}] published")
        prev_stamp = t_sec
        rate.sleep()

    rospy.loginfo("✅ 완료")
    rospy.signal_shutdown("done")

def gt_pose_to_imu(poses, timestamps, i):
    """GT pose에서 IMU 데이터 생성 (안정적인 버전)"""
    if i == 0:
        # 첫 번째 프레임은 정지 상태로 가정
        return (1, 0, 0, 0), (0, 0, 9.81), (0, 0, 0)
    
    # 현재와 이전 pose
    T_curr = poses[i]
    T_prev = poses[i-1]
    
    # 시간 간격
    dt = timestamps[i] - timestamps[i-1]
    if dt <= 0:
        dt = 0.1  # 기본값
    
    # 쿼터니언 (현재 pose에서)
    quat = quaternion_from_matrix(T_curr)
    
    # 가속도 계산 (더 안정적인 방식)
    if i > 2:  # 더 많은 프레임 사용
        # 3점 평균으로 스무딩
        pos_curr = T_curr[:3, 3]
        pos_prev = T_prev[:3, 3]
        pos_prev2 = poses[i-2][:3, 3]
        pos_prev3 = poses[i-3][:3, 3]
        
        # 중앙 차분으로 속도 계산
        velocity_curr = (pos_curr - pos_prev) / dt
        velocity_prev = (pos_prev - pos_prev2) / dt
        velocity_prev2 = (pos_prev2 - pos_prev3) / dt
        
        # 가속도 계산 (중앙 차분)
        acceleration = (velocity_curr - velocity_prev2) / (2 * dt)
        
        # 노이즈 필터링
        acceleration = np.clip(acceleration, -5, 5)  # 더 보수적인 제한
        
        # 중력 추가 (Z축 방향)
        acceleration[2] += 9.81
        
    else:
        acceleration = np.array([0, 0, 9.81])  # 중력만
    
    # 각속도 계산 (간단하고 안정적인 방식)
    if i > 1:
        # 쿼터니언 차분으로 각속도 계산
        quat_prev = quaternion_from_matrix(T_prev)
        
        # 쿼터니언 차분 (numpy 배열로 처리)
        # quat = [w, x, y, z] 형식
        q1_w, q1_x, q1_y, q1_z = quat
        q2_w, q2_x, q2_y, q2_z = quat_prev
        
        # 쿼터니언 곱셈 (q1 * q2_conjugate)
        q_diff_w = q1_w * q2_w + q1_x * q2_x + q1_y * q2_y + q1_z * q2_z
        q_diff_x = q1_w * q2_x - q1_x * q2_w - q1_y * q2_z + q1_z * q2_y
        q_diff_y = q1_w * q2_y + q1_x * q2_z - q1_y * q2_w - q1_z * q2_x
        q_diff_z = q1_w * q2_z - q1_x * q2_y + q1_y * q2_x - q1_z * q2_w
        
        # 각속도 추출
        angle = 2 * np.arccos(np.clip(abs(q_diff_w), 0, 1))
        if angle > 1e-6:
            axis = np.array([q_diff_x, q_diff_y, q_diff_z]) / np.sin(angle/2)
            angular_velocity = axis * angle / dt
        else:
            angular_velocity = np.array([0, 0, 0])
        
        # 각속도 필터링
        angular_velocity = np.clip(angular_velocity, -2, 2)  # 더 보수적
    else:
        angular_velocity = np.array([0, 0, 0])
    
    return quat, tuple(acceleration), tuple(angular_velocity)

if __name__=="__main__":
    seq="2013_05_28_drive_0000_sync"
    base="/home/donok/kitti360"
    lidar_dir      = f"{base}/KITTI-360/data_3d_raw/{seq}/velodyne_points/data"
    imu_dir        = f"{base}/KITTI-360/data_poses_oxts/data_poses/{seq}/oxts/data"
    ts_path        = f"{base}/KITTI-360/data_3d_raw/{seq}/velodyne_points/timestamps.txt"
    gt_path        = f"{base}/KITTI-360/data_poses/{seq}/poses.txt"
    T_cam0_lidar   = np.eye(4)  # calib_cam_to_velo.txt 대로 수정

    # GT pose에서 IMU 생성 모드 (True: GT에서 생성, False: 기존 IMU 파일 사용)
    use_gt_imu = True  # 개선된 GT IMU 테스트

    # rosbag 녹화 - 새로운 영어 레이블 사용
    now = datetime.now().strftime("%Y%m%d_%H%M")
    mode_str = "gt_imu" if use_gt_imu else "real_imu"
    bag = f"/media/donok/EXTERNAL_DRIVE/bagfile/{mode_str}_{seq}_{now}.bag"  # 기존 bagfile 폴더 사용
    
    # 디렉토리가 없으면 생성
    import os
    os.makedirs(os.path.dirname(bag), exist_ok=True)
    
    rosbag_proc = subprocess.Popen([
        "rosbag", "record", "-O", bag,
        "/velodyne/points", "/imu/data", "/gt_pose"
    ])
    try:
        publish_all(lidar_dir, imu_dir, ts_path, gt_path, T_cam0_lidar, use_gt_imu)
    finally:
        rospy.loginfo("⛔ 퍼블리셔 종료")
        rosbag_proc.terminate()
