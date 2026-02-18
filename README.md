# VACOP Mapping & Localization (ROS 2)

Ce dépôt contient les configurations de cartographie et de localisation pour le robot **VACOP**, utilisant **RTAB-Map** et un Lidar **Robosense (RS-Lidar)**.

## État du projet : Géoréférencement
Bien que les fichiers `rtabmap_lidar_launch_icp_imu_gnss.py` et `rtabmap_lidar_launch_icp_imu_gnss_referencer.py` contiennent des paramètres pour le **GPS/GNSS**, cette fonctionnalité est actuellement **expérimentale et non fonctionnelle**.
* Les topics GPS sont configurés, mais le système ne valide pas encore les contraintes de position globale.
* Le robot s'appuie actuellement sur l'odométrie ICP et l'IMU pour sa navigation.

---

## Dépendances
- ROS 2 (Humble/Foxy)
- `rtabmap_ros`
- `rslidar_sdk` (Driver Robosense)
- `kiss-icp` (optionnel)

---

## Fichiers de Lancement (Launch Files)

| Fichier | Description |
| :--- | :--- |
| `rtabmap_lidar_launch.py` | **Version stable** : Utilise le Lidar seul (ICP) pour l'odométrie et la carte. |
| `rtabmap_lidar_launch_icp_imu_gnss.py` | Utilise **KISS-ICP** pour une odométrie lidar plus robuste. |
| `rtabmap_lidar_launch_icp_imu_gnss_referencer.py` | Version dédiée aux tests de recalage via coordonnées GPS. |

---

## Explication des Paramètres Techniques

Voici la liste complète des paramètres présents dans les scripts de lancement :

### 1. Configuration de l'Odométrie (Nœud `icp_odometry`)
* **`frame_id`** : Identifiant du repère de référence du robot (ex: `base_link` ou `rslidar`).
* **`odom_frame_id`** : Nom du repère de l'odométrie (généralement `odom`).
* **`publish_tf`** : Si `true`, le nœud publie la transformation entre `odom` et `base_link`.
* **`wait_for_transform`** : Temps d'attente (secondes) pour qu'une transformation TF soit disponible.
* **`approx_sync`** : Autorise la synchronisation des données (Lidar/IMU) même si leurs horodatages ne sont pas identiques.
* **`queue_size`** : Nombre de messages stockés en mémoire avant d'être traités.
* **`subscribe_imu`** : Si `true`, le nœud utilise l'IMU pour assister le calcul du mouvement.
* **`wait_imu_to_init`** : Attend de recevoir des données IMU valides avant de commencer le calcul.

### 2. Algorithme ICP (Iterative Closest Point)
* **`Icp/VoxelSize` (0.35)** : Taille de la grille de filtrage. Réduit la densité des points pour stabiliser le calcul face au bruit (herbe, feuilles).
* **`Icp/PointToPlane`** : Utilise l'algorithme "point-à-plan" pour une meilleure précision sur les surfaces planes (murs, sol).
* **`Icp/PointToPlaneK` (20)** : Nombre de points voisins utilisés pour estimer la normale d'une surface.
* **`Icp/MaxCorrespondenceDistance` (1.0)** : Distance maximale (mètres) tolérée entre deux scans pour qu'ils soient alignés. Crucial pour les mouvements rapides.
* **`Icp/Iterations` (50)** : Nombre maximum d'essais pour aligner deux scans.
* **`Icp/Epsilon` (0.001)** : Seuil de précision pour arrêter les itérations de l'algorithme.
* **`Reg/Force3DoF`** : Bloque les axes Z, Roll et Pitch. Le robot est considéré comme restant sur un plan 2D parfait.
* **`Odom/GuessMotion`** : Utilise la vitesse calculée précédemment pour prédire la position suivante.
* **`Odom/ResetCountdown` (1)** : Réinitialise l'odométrie après X scans si elle est perdue.

### 3. Configuration du SLAM (Nœud `rtabmap`)
* **`subscribe_depth` / `subscribe_rgb` / `subscribe_stereo`** : Activés/Désactivés pour ignorer les caméras et se concentrer sur le Lidar.
* **`subscribe_scan_cloud`** : Active l'entrée PointCloud2 du Lidar.
* **`subscribe_odom` / `odom_topic`** : Indique au SLAM d'écouter une odométrie externe (ex: `/odom` ou `/kiss/odometry`).
* **`Mem/IncrementalMemory`** : Si `true`, RTAB-Map continue d'apprendre et de construire la carte.
* **`Rtabmap/DetectionRate`** : Fréquence (Hz) de mise à jour de la carte.
* **`RGBD/ProximityBySpace`** : Détecte les boucles locales si le robot repasse près d'un endroit connu géographiquement.

### 4. Paramètres de Carte (Grid)
* **`Grid/CellSize` (0.1)** : Résolution de la carte 2D (10 cm par pixel).
* **`Grid/RangeMax` (90.0)** : Distance maximale à laquelle les points Lidar sont intégrés à la carte.

### 5. Géoréférencement (Actuellement inactifs)
* **`subscribe_gps`** : Active l'écoute du topic GPS.
* **`Rtabmap/LoopGPS`** : Utilise le GPS pour détecter les fermetures de boucle.
* **`Rtabmap/GeoTargeting`** : Aligne la carte sur des coordonnées géographiques réelles.
* **`Optimizer/PriorKnowledge`** : Utilise les positions GPS comme contraintes fortes pour l'optimisation de la carte.

---

## Lancement

Pour lancer la version stable (Lidar ICP) :
```bash
ros2 launch vacop_mapping rtabmap_lidar_launch.py
