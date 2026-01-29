from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config = get_package_share_directory('rslidar_sdk') + '/rviz/rviz2.rviz'
    config_file = '' # laisse vide pour le défaut

    return LaunchDescription([
        Node(
            namespace='rslidar_sdk', 
            package='rslidar_sdk', 
            executable='rslidar_sdk_node', 
            output='screen', 
            parameters=[{'config_path': config_file}]
        ),
        
        # Node(
        #     namespace='rviz2', 
        #     package='rviz2', 
        #     executable='rviz2', 
        #     arguments=['-d', rviz_config]
        # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'rslidar']
        )
    ])
