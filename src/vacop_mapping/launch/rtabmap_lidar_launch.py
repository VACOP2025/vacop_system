import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- PARTIE 1 : Lancer le Driver du Lidar ---
    # On va chercher le fichier de lancement du driver dans rslidar_sdk
    # Note : Vérifie si le fichier s'appelle 'start.py' ou 'start.launch.py' dans rslidar_sdk/launch
    try:
        rslidar_dir = get_package_share_directory('rslidar_sdk')
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rslidar_dir, 'launch', 'start.py') 
            )
        )
    except Exception as e:
        print(f"ATTENTION: Impossible de trouver rslidar_sdk. Erreur: {e}")
        # On continue quand même pour tester le mapping si on joue un bag, sinon ça plantera.
        lidar_launch = Node(package='dummy', executable='dummy') # Placeholder

    # --- PARTIE 2 : Ton Code SLAM Original ---
    
    frame_id = 'rslidar' 
    use_sim_time = False 
    
    # 1. Nœud d'ODOMÉTRIE (Optimisé pour Robot Mobile Extérieur)
    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        output='screen',
        parameters=[{
            #'use_sim_time': use_sim_time,
            'frame_id': frame_id,
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'queue_size': 20,
            
            # --- Réglages ROBUSTES pour Extérieur ---
            
            # Filtrage et Précision
            'Icp/VoxelSize': '0.35',          ### Augmenté (0.1 -> 0.2) pour réduire le bruit (herbe/feuilles) et le CPU
            'Icp/PointToPlane': 'true',
            'Icp/PointToPlaneK': '20',       ### AJOUTÉ : Nécessaire pour bien calculer les surfaces (murs/sol)
            
            # Tolérance de mouvement (Vitesse)
            'Icp/MaxCorrespondenceDistance': '1', ### AJOUTÉ : (0.1 -> 0.6) Crucial ! Tolère 60cm de décalage entre 2 scans
            'Icp/Iterations': '50',          ### Augmenté (10 -> 30) pour laisser le calcul converger
            'Icp/Epsilon': '0.001',          ### AJOUTÉ : Seuil de précision d'arrêt
            
            # Contraintes Physiques du Robot
            'Reg/Force3DoF': 'true',         ### AJOUTÉ : Bloque Z, Roll, Pitch. Le robot reste collé au sol.
            'Odom/GuessMotion': 'true',      ### AJOUTÉ : Utilise la vitesse précédente pour anticiper la suite
            'Odom/ResetCountdown': '1'       ### AJOUTÉ : Si perdu, tente de reset automatiquement
        }],
        remappings=[
            ('scan_cloud', '/rslidar_points')
        ]
    )

    # 2. Nœud SLAM (Construit la carte et la boucle)
    slam_node = Node(
        package='rtabmap_slam', 
        executable='rtabmap', 
        output='screen',
        parameters=[{
            #'use_sim_time': use_sim_time,
            'frame_id': frame_id,
            'map_frame_id': 'map',
            
            # --- Entrées ---
            'subscribe_rgb': False,
            'subscribe_depth': False,       
            'subscribe_scan': False,
            'subscribe_scan_cloud': True,
            'subscribe_odom_info': True,
            
            # --- Performance ---
            'approx_sync': True,
            'queue_size': 20,
            'wait_for_transform': 0.2,
            
            # --- Mapping ---
            'Mem/IncrementalMemory': 'true', # Mode SLAM
            'Mem/InitWMWithAllNodes': 'false',
            'Rtabmap/DetectionRate': '1.0',
            'Grid/RangeMax': '90.0',         # Portée max pour la carte 2D (optionnel)
            'Grid/CellSize': '0.1',         # Taille des cases de la grille
        }],
        remappings=[
            ('scan_cloud', '/rslidar_points'),
            ('rgb/image', '/ignored_rgb'),
            ('depth/image', '/ignored_depth'),
            ('imu', '/ignored_imu')
        ]
    )

    # 3. Visualisation
    viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[{
            #'use_sim_time': use_sim_time,
            'frame_id': frame_id,
            'subscribe_odom_info': True,
            'subscribe_scan_cloud': True,
            'approx_sync': True,
            'queue_size': 20
            # wait_for_transform crash
        }],
        remappings=[
            ('scan_cloud', '/rslidar_points')
        ]
    )

    return LaunchDescription([
        lidar_launch,      # <-- Lance le Lidar
        icp_odometry_node, # <-- Lance l'odometrie
        slam_node,         # <-- Lance le SLAM
        viz_node           # <-- Lance la Visu
    ])
