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

# === 설정 ===
POSE_FILE = "/home/donok/kitti360/KITTI-360/data_poses/2013_05_28_drive_0000_sync/poses.txt"
TIMESTAMP_FILE = "/home/donok/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/timestamps.txt"
BIN_DIR = "/home/donok/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data"
OXTS_DIR = "/home/donok/kitti360/KITTI-360/data_poses_oxts/data_poses/2013_05_28_drive_0000_sync/oxts/data"
OXTS_TIMESTAMP_FILE = "/home/donok/kitti360/KITTI-360/data_poses_oxts/data_poses/2013_05_28_drive_0000_sync/oxts/timestamps.txt"
BAG_FILE = "/media/donok/EXTERNAL_DRIVE/bagfile/gt_lio_input.bag"
FRAME_ID = "base_link"  # LIO-SAM에서 사용하는 frame ID로 맞춰야 함

def read_timestamps(path):
    timestamps = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        base_line = lines[0].split(".")
        base_time = datetime.strptime(base_line[0] + "." + base_line[1][:6], "%Y-%m-%d %H:%M:%S.%f").timestamp()  # float 초 단위
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
    header.stamp = rospy.Time.from_sec(timestamp)
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
    from scipy.spatial.transform import Rotation as R
    trans = T[:3, 3]
    rot = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w

    msg = Odometry()
    msg.header.stamp = rospy.Time.from_sec(timestamp)
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

def oxts_to_imu(oxts_data, timestamp):
    """OXTS 데이터를 IMU 메시지로 변환"""
    from scipy.spatial.transform import Rotation as R
    
    msg = Imu()
    msg.header.stamp = rospy.Time.from_sec(timestamp)
    msg.header.frame_id = FRAME_ID
    
    # OXTS 데이터 파싱 (dataformat.txt 참조)
    # ax, ay, az: 가속도 (m/s^2)
    ax, ay, az = oxts_data[11], oxts_data[12], oxts_data[13]
    msg.linear_acceleration.x = ax
    msg.linear_acceleration.y = ay
    msg.linear_acceleration.z = az
    
    # wx, wy, wz: 각속도 (rad/s)
    wx, wy, wz = oxts_data[17], oxts_data[18], oxts_data[19]
    msg.angular_velocity.x = wx
    msg.angular_velocity.y = wy
    msg.angular_velocity.z = wz
    
    # roll, pitch, yaw: 오일러 각도 (rad)
    roll, pitch, yaw = oxts_data[3], oxts_data[4], oxts_data[5]
    # 오일러 각도를 쿼터니언으로 변환
    rot = R.from_euler('xyz', [roll, pitch, yaw])
    quat = rot.as_quat()
    msg.orientation.x = quat[0]
    msg.orientation.y = quat[1]
    msg.orientation.z = quat[2]
    msg.orientation.w = quat[3]
    
    return msg

def main():
    rospy.init_node("gt_bag_generator", anonymous=True)
    base_time, timestamps = read_timestamps(TIMESTAMP_FILE)
    poses = read_poses(POSE_FILE)
    
    # OXTS 데이터 읽기
    oxts_base_time, oxts_timestamps = read_oxts_timestamps(OXTS_TIMESTAMP_FILE)
    oxts_data = read_oxts_data(OXTS_DIR)

    bin_files = sorted(os.listdir(BIN_DIR))
    bag = rosbag.Bag(BAG_FILE, 'w')

    for i, bin_name in enumerate(bin_files):
        idx = int(os.path.splitext(bin_name)[0])
        if idx not in poses:
            print(f"[skip] {bin_name} - no GT pose")
            continue

        bin_path = os.path.join(BIN_DIR, bin_name)
        timestamp_ros = base_time + timestamps[idx]  # ⬅️ ROS 시간 보정됨
        T = poses[idx]

        pc_msg = bin_to_pointcloud2(bin_path, timestamp_ros)
        odom_msg = pose_to_odometry(T, timestamp_ros)
        
        # OXTS 데이터에서 IMU 생성 (인덱스 매칭 필요)
        if idx < len(oxts_data):
            imu_msg = oxts_to_imu(oxts_data[idx], timestamp_ros)
            bag.write('/imu_correct', imu_msg, t=imu_msg.header.stamp)

        bag.write('/velodyne_points', pc_msg, t=pc_msg.header.stamp)
        bag.write('/odometry/imu', odom_msg, t=odom_msg.header.stamp)

        print(f"[write] idx={idx}, time={timestamp_ros:.6f}")

    bag.close()
    print(f"\n✅ Done: saved {BAG_FILE}")


if __name__ == '__main__':
    main()
