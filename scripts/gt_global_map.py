import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

# 사용자 설정
pose_file = "/home/donok/kitti360/KITTI-360/data_poses/2013_05_28_drive_0000_sync/poses.txt"  # 4x4 pose 행렬이 한 줄당 12개 (3x4 행렬)로 저장된 파일
bin_dir = "/home/donok/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data"  # .bin 파일 경로

T_lidar_to_body = np.eye(4)  # 필요한 경우 Extrinsic 정보 입력

def load_poses_with_index(path):
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            idx = int(parts[0])
            pose = np.eye(4)
            pose[:3, :4] = np.array(parts[1:]).reshape(3, 4)
            poses[idx] = pose
    return poses

def load_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def main():
    poses = load_poses_with_index(pose_file)
    pcd_all = o3d.geometry.PointCloud()

    for idx in tqdm(sorted(poses.keys())):
        bin_filename = f"{idx:010d}.bin"
        bin_path = os.path.join(bin_dir, bin_filename)
        if not os.path.exists(bin_path):
            print(f"⚠️ {bin_path} 없음, 건너뜀")
            continue

        pose = poses[idx] @ T_lidar_to_body
        pts = load_bin(bin_path)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_world = (pose @ pts_h.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd_all += pcd

    print("다운샘플링 중...")
    pcd_all = pcd_all.voxel_down_sample(voxel_size=0.1)
    o3d.io.write_point_cloud("global_map.pcd", pcd_all)
    print("✅ global_map.pcd 저장 완료")

    o3d.visualization.draw_geometries([pcd_all])

if __name__ == "__main__":
    main()
