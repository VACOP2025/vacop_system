import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    # --- PARTIE 1 : Driver Lidar ---
    # On force le chemin puisque AMENT ne le voit pas
    rslidar_dir = '/root/vacop_ws/install/rslidar_sdk/share/rslidar_sdk'
    
    if os.path.exists(rslidar_dir):
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rslidar_dir, 'launch', 'start.py') 
            )
        )
        ld.add_action(lidar_launch)
        print("INFO: Driver rslidar_sdk chargé manuellement depuis le chemin install.")
    else:
        ld.add_action(LogInfo(msg="ATTENTION: Chemin rslidar_sdk introuvable même en manuel."))

    # --- PARTIE 2 : Odométrie ICP ---
    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'queue_size': 20,
            'Icp/VoxelSize': '0.35',
            'Icp/PointToPlane': 'true',
            'Icp/PointToPlaneK': '20',
            'Icp/MaxCorrespondenceDistance': '1.0',
            'Icp/Iterations': '50',
            'Reg/Force3DoF': 'true',
            'Odom/GuessMotion': 'true'
        }],
        remappings=[('scan_cloud', '/rslidar_points')]
    )

    # --- PARTIE 3 : SLAM rtabmap ---
    slam_node = Node(
        package='rtabmap_slam', 
        executable='rtabmap', 
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'map_frame_id': 'map',
            'subscribe_rgb': False,
            'subscribe_depth': False,       
            'subscribe_scan_cloud': True,
            'approx_sync': True,
            'Grid/RangeMax': '90.0',
            'Mem/IncrementalMemory': 'true',
        }],
        remappings=[('scan_cloud', '/rslidar_points')]
    )

    ld.add_action(icp_odometry_node)
    ld.add_action(slam_node)
    
    return ld