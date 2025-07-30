#!/usr/bin/env python3
import rosbag
import rospy
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, TwistWithCovariance, Vector3
import os
from std_msgs.msg import Header
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# === 설정 ===
POSE_FILE = "/home/donok/kitti360/KITTI-360/data_poses/2013_05_28_drive_0000_sync/poses.txt"
TIMESTAMP_FILE = "/home/donok/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/timestamps.txt"
BIN_DIR = "/home/donok/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data"
OXTS_DIR = "/home/donok/kitti360/KITTI-360/data_poses_oxts/data_poses/2013_05_28_drive_0000_sync/oxts/data"
OXTS_TIMESTAMP_FILE = "/home/donok/kitti360/KITTI-360/data_poses_oxts/data_poses/2013_05_28_drive_0000_sync/oxts/timestamps.txt"
BAG_FILE = "/media/donok/EXTERNAL_DRIVE/bagfile/gt_corrected_kitti360.bag"
FRAME_ID = "base_link"
MAX_FRAMES = 100000  # 전체 데이터로 설정 (충분히 큰 값)

def read_timestamps(path):
    timestamps = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        base_line = lines[0].split(".")
        base_time = datetime.strptime(base_line[0] + "." + base_line[1][:6], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        for line in lines:
            parts = line.split(".")
            time_str = parts[0] + "." + parts[1][:6]
            current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            delta = current_time - base_time
            timestamps.append(delta)
    return base_time, timestamps

def read_poses(path):
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            items = line.strip().split()
            idx = int(items[0])
            T = np.eye(4)
            T[:3, :4] = np.array([float(x) for x in items[1:]]).reshape(3, 4)
            poses[idx] = T
    return poses

def read_oxts_data(oxts_dir):
    """OXTS 데이터 읽기"""
    oxts_data = {}
    oxts_files = sorted(os.listdir(oxts_dir))
    
    for i, filename in enumerate(oxts_files):
        filepath = os.path.join(oxts_dir, filename)
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            items = line.split()
            if len(items) >= 30:  # OXTS 데이터는 30개 필드
                oxts_data[i] = [float(x) for x in items]
    
    return oxts_data

def read_oxts_timestamps(path):
    """OXTS 타임스탬프 읽기"""
    timestamps = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        base_line = lines[0].split(".")
        base_time = datetime.strptime(base_line[0] + "." + base_line[1][:6], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        for line in lines:
            parts = line.split(".")
            time_str = parts[0] + "." + parts[1][:6]
            current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            delta = current_time - base_time
            timestamps.append(delta)
    return base_time, timestamps

def find_closest_oxts_index(target_time, oxts_timestamps):
    """가장 가까운 OXTS 인덱스 찾기"""
    min_diff = float('inf')
    closest_idx = 0
    for i, oxts_time in enumerate(oxts_timestamps):
        diff = abs(target_time - oxts_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    return closest_idx

def gt_corrected_imu(gt_poses, timestamps, idx, prev_idx=None):
    """GT 포즈로부터 보정된 IMU 데이터 생성"""
    msg = Imu()
    # 타임스탬프를 현재 시간 기준으로 설정
    current_time = rospy.Time.now()
    msg.header.stamp = current_time
    msg.header.frame_id = FRAME_ID
    
    T = gt_poses[idx]
    
    # GT 포즈로부터 각속도 계산
    if prev_idx is not None and prev_idx in gt_poses:
        T_prev = gt_poses[prev_idx]
        dt = timestamps[idx] - timestamps[prev_idx]
        
        # 회전 행렬의 차이로 각속도 계산
        R_prev = R.from_matrix(T_prev[:3, :3])
        R_curr = R.from_matrix(T[:3, :3])
        R_diff = R_curr * R_prev.inv()
        
        # 각속도 계산
        euler_diff = R_diff.as_euler('xyz')
        angular_vel = Vector3()
        angular_vel.x = euler_diff[0] / dt if dt > 0 else 0.0
        angular_vel.y = euler_diff[1] / dt if dt > 0 else 0.0
        angular_vel.z = euler_diff[2] / dt if dt > 0 else 0.0
        msg.angular_velocity = angular_vel
        
        # 가속도 계산 (위치의 이차미분)
        pos_curr = T[:3, 3]
        pos_prev = T_prev[:3, 3]
        pos_prev2 = gt_poses.get(prev_idx-1, T_prev)[:3, 3] if prev_idx-1 in gt_poses else pos_prev
        
        # 중앙차분법으로 가속도 계산
        vel_curr = (pos_curr - pos_prev) / dt if dt > 0 else np.zeros(3)
        vel_prev = (pos_prev - pos_prev2) / dt if dt > 0 else np.zeros(3)
        acc = (vel_curr - vel_prev) / dt if dt > 0 else np.zeros(3)
        
        msg.linear_acceleration.x = acc[0]
        msg.linear_acceleration.y = acc[1]
        msg.linear_acceleration.z = acc[2] + 9.81  # 중력 추가
    else:
        # 첫 번째 프레임
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 9.81
    
    # 방향 (쿼터니언)
    rot = R.from_matrix(T[:3, :3]).as_quat()
    msg.orientation.x = rot[0]
    msg.orientation.y = rot[1]
    msg.orientation.z = rot[2]
    msg.orientation.w = rot[3]
    
    return msg

def bin_to_pointcloud2(bin_path, timestamp, scan_duration=0.1):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    N = points.shape[0]

    # 거리 기반 필터링 (노이즈 제거)
    distances = np.linalg.norm(points[:, :3], axis=1)
    valid_mask = (distances > 1.0) & (distances < 100.0)  # 1m ~ 100m 범위만 사용
    points = points[valid_mask]
    N = points.shape[0]

    # 수직 각도 계산
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    vertical_angle = np.arctan2(points[:, 2], xy_norm) * 180 / np.pi  # in degrees

    # KITTI-360 기준: -24.8도 ~ 2.0도 범위, 0.4도 해상도
    min_angle = -24.8
    ang_res = 0.4
    ring = np.floor((vertical_angle - min_angle) / ang_res).astype(np.uint16)
    ring = np.clip(ring, 0, 63).reshape(-1, 1)  # clip to [0, 63]

    # 상대 시간 필드 생성
    rel_time = np.linspace(0.0, scan_duration, N).reshape(-1, 1).astype(np.float32)

    # 최종 포인트 구성
    points_all = np.hstack([points, ring, rel_time])

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.UINT16, 1),
        PointField('time', 18, PointField.FLOAT32, 1),
    ]

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('ring', np.uint16),
        ('time', np.float32),
    ])
    structured = np.array([tuple(row) for row in points_all], dtype=dtype)
    data = structured.tobytes()

    header = Header()
    # 타임스탬프를 현재 시간 기준으로 설정
    current_time = rospy.Time.now()
    header.stamp = current_time
    header.frame_id = FRAME_ID

    pc2 = PointCloud2(
        header=header,
        height=1,
        width=N,
        fields=fields,
        is_bigendian=False,
        point_step=22,
        row_step=22 * N,
        is_dense=True,
        data=data
    )
    return pc2

def pose_to_odometry(T, timestamp):
    trans = T[:3, 3]
    rot = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w

    msg = Odometry()
    # 타임스탬프를 현재 시간 기준으로 설정
    current_time = rospy.Time.now()
    msg.header.stamp = current_time
    msg.header.frame_id = "map"
    msg.child_frame_id = FRAME_ID
    msg.pose.pose.position.x = trans[0]
    msg.pose.pose.position.y = trans[1]
    msg.pose.pose.position.z = trans[2]
    msg.pose.pose.orientation.x = rot[0]
    msg.pose.pose.orientation.y = rot[1]
    msg.pose.pose.orientation.z = rot[2]
    msg.pose.pose.orientation.w = rot[3]
    return msg

def main():
    rospy.init_node("gt_corrected_bag_generator", anonymous=True)
    
    print("Loading data...")
    base_time, timestamps = read_timestamps(TIMESTAMP_FILE)
    poses = read_poses(POSE_FILE)
    
    # OXTS 데이터 읽기 (참고용)
    oxts_base_time, oxts_timestamps = read_oxts_timestamps(OXTS_TIMESTAMP_FILE)
    oxts_data = read_oxts_data(OXTS_DIR)
    
    print(f"Loaded {len(poses)} poses, {len(oxts_data)} OXTS data points")
    
    bin_files = sorted(os.listdir(BIN_DIR))
    bag = rosbag.Bag(BAG_FILE, 'w')

    processed_frames = 0
    prev_idx = None
    
    for i, bin_name in enumerate(bin_files):
        if processed_frames >= MAX_FRAMES:
            print(f"[stop] Reached {MAX_FRAMES} frames limit")
            break
            
        idx = int(os.path.splitext(bin_name)[0])
        if idx not in poses:
            print(f"[skip] {bin_name} - no GT pose")
            continue

        bin_path = os.path.join(BIN_DIR, bin_name)
        timestamp_ros = base_time + timestamps[idx]
        T = poses[idx]

        # GT 보정된 데이터 생성
        pc_msg = bin_to_pointcloud2(bin_path, timestamp_ros)
        odom_msg = pose_to_odometry(T, timestamp_ros)
        imu_msg = gt_corrected_imu(poses, timestamps, idx, prev_idx)

        # Bag에 쓰기 (타임스탬프를 순차적으로 증가)
        bag_time = rospy.Time.from_sec(processed_frames * 0.1)  # 10Hz로 가정
        bag.write('/velodyne_points', pc_msg, t=bag_time)
        bag.write('/odometry/imu', odom_msg, t=bag_time)
        bag.write('/imu_correct', imu_msg, t=bag_time)

        prev_idx = idx
        processed_frames += 1
        print(f"[write] idx={idx}, time={timestamp_ros:.6f} ({processed_frames}/{MAX_FRAMES})")

    bag.close()
    print(f"\n✅ Done: saved {BAG_FILE}")

if __name__ == '__main__':
    main() 