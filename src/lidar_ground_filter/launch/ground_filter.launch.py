from launch import LaunchDescription
from launch_ros.actions import Node

#A MODIFIER SELON LE TOPIC SUR LA JETSON
POINTCLOUD_TOPIC_NAME = '/rslidar_points'


def generate_launch_description():
    # A MODIFIER POUR AVOIR LA TRANFORMER ENTRE LA BASE (LE SOL AU MILIEU DE L'ESSIEU ARRIÈRE) ET LE LIDAR
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link',
        arguments=['0.73', '0', '1.2', '0', '0.0174533', '0.0523599', 'base_link', 'rslidar']
    )

    ground_filter = Node(
        package='lidar_ground_filter',
        executable='ransac_ground_filter_node',
        name='ransac_ground_filter',
        output='screen',
        parameters=[{
            'base_frame': 'base_link',
            'voxel_size': 0.1,
            'distance_threshold': 0.10, # A RÉGLER POUR ENLEVER PLUS OU MOINS LES TROTTOIRS
            'plane_slope_threshold': 15.0,
        }],
        remappings=[
            ('input/points', POINTCLOUD_TOPIC_NAME)
        ]
    )

    clustering = Node(
        package='lidar_ground_filter',
        executable='euclidean_cluster_node',
        name='euclidean_cluster',
        output='screen',
        parameters=[{
            'cluster_tolerance': 0.5,  # POUR DÉFINIR SI DES POINTS ÉLOIGNÉS SONT LE MÊME OBJET (ICI 50 CM)
            'min_cluster_size': 10,    # IGNORE LE BRUIT (MOINS DE 10 CLUSTERS)
        }],
        remappings=[
            ('input/no_ground', 'perception/scan/no_ground') 
        ]
    )

    return LaunchDescription([
        lidar_tf,
        ground_filter
        #clustering
    ])
