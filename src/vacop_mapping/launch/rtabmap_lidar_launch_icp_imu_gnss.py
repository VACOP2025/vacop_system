import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- CONFIGURATION DES TOPICS ---
    lidar_topic = '/rslidar_points'
    gps_topic   = '/gps/fix'
    imu_topic   = '/imu/data'      
    
    # --- CONFIGURATION DES FRAMES ---
    robot_frame = 'base_link' 
    odom_frame  = 'odom'
    
    # --- PARTIE 1 : Driver Lidar ---
    try:
        rslidar_dir = get_package_share_directory('rslidar_sdk')
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rslidar_dir, 'launch', 'start.py') 
            )
        )
    except Exception as e:
        print(f"ATTENTION: Impossible de trouver rslidar_sdk. Erreur: {e}")
        lidar_launch = Node(package='dummy', executable='dummy') 

    # --- PARTIE 2 : TF STATIQUES (Position des capteurs) ---
    
    # 1. Position GPS (Ex: 0,0,0 par défaut)
    gps_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='gps_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'gps_link']
    )
    
    # 2. Position IMU (NOUVEAU)
    # Important : Si l'IMU est tourné, ajustez le 'yaw pitch roll' (les 3 derniers 0)
    # Arguments: x y z yaw pitch roll parent child
    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link']
    )

    # --- PARTIE 3 : KISS-ICP (Odométrie + IMU) ---
    kiss_icp_node = Node(
        package='kiss_icp',
        executable='kiss_icp_node',
        name='kiss_icp_node',
        output='screen',
        parameters=[{
            'topic': lidar_topic,
            'visualize': False,
            'deskew': True,
            'max_range': 50.0,
            'min_range': 2.0,
            'base_frame': robot_frame,
            'odom_frame': odom_frame,
            'publish_odom_tf': True,

        }],
        remappings=[
            ('pointcloud_topic', lidar_topic),
            ('odometry', '/kiss/odometry'),
            ('imu', imu_topic)
        ]
    )

    # --- PARTIE 4 : RTAB-Map (SLAM + GPS + IMU) ---
    slam_node = Node(
        package='rtabmap_slam', 
        executable='rtabmap', 
        output='screen',
        arguments=['--delete_db_on_start'], 
        parameters=[{
            'frame_id': robot_frame,      
            'map_frame_id': 'map',
            'odom_frame_id': odom_frame,  
            
            # Entrées
            'subscribe_odom': True,
            'odom_topic': '/kiss/odometry', 
            'subscribe_scan_cloud': True,
            'approx_sync': True,
            'sync_queue_size': 20,
            'topic_queue_size': 10,
            
            # --- GPS ---
            'gps_topic': gps_topic,
            'Rtabmap/LoopGPS': 'true',
            
            # --- IMU ---
            'wait_imu_to_init': True,
            
            # Tuning SLAM
            'Reg/Strategy': '1',             
            'Reg/Force3DoF': 'true',         
            'RGBD/ProximityBySpace': 'true',
            'RGBD/LinearUpdate': '0.2',
            'Mem/IncrementalMemory': 'true', 
            'Grid/RangeMax': '50.0',
            'Grid/CellSize': '0.1',
        }],
        remappings=[
            ('scan_cloud', lidar_topic),
            ('rgb/image', '/ignored_rgb'),
            ('depth/image', '/ignored_depth'),
            ('rgb/camera_info', '/ignored_camera_info'),
            ('gps/fix', gps_topic),
            ('imu', imu_topic)
        ]
    )

    # --- PARTIE 5 : Visualisation ---
    viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[{
            'frame_id': robot_frame,
            'odom_frame_id': odom_frame,
            'subscribe_odom': True,
            'subscribe_scan_cloud': True,
            'odom_topic': '/kiss/odometry',
            'approx_sync': True,
            'sync_queue_size': 20
        }],
        remappings=[
            ('scan_cloud', lidar_topic),
            ('rgb/image', '/ignored_rgb'),
            ('depth/image', '/ignored_depth'),
            ('rgb/camera_info', '/ignored_camera_info')
        ]
    )

    return LaunchDescription([
        lidar_launch,
        gps_tf,
        imu_tf,
        kiss_icp_node,
        slam_node,
        viz_node
    ])