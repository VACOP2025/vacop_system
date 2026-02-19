# Calibration extrinsèque LiDAR → Caméra

## Objectif

Ce script réalise le calibrage extrinsèque entre un LiDAR et une caméra
à partir d'un jeu de données contenant :

-   Les paramètres intrinsèques caméra (`calibration.yml`)
-   Des images du damier
-   Des nuages de points LiDAR correspondants

L'objectif est d'estimer la transformation rigide :

    T_lidar_to_camera

permettant de projeter les points LiDAR dans l'image caméra.

------------------------------------------------------------------------

## Données requises

Structure attendue :

    dataset_test/
        calibration.yml
        images/
            0001.jpg
            ...
        pointclouds/
            0001.asc
            ...

------------------------------------------------------------------------

## Configuration géométrique de la mire

-   Damier : 6 × 8 cases
-   Coins internes : 5 × 7
-   Taille d'une case : 0.10 m
-   Bord blanc : 0.10 m autour
-   Dimensions totales panneau : 0.80 m × 1.00 m

------------------------------------------------------------------------

## Méthodologie

### 1. Pose de la mire dans la caméra

-   Détection des coins internes (`findChessboardCornersSB`)
-   Estimation de la pose via `solvePnP`
-   Obtention de T_board_to_camera

------------------------------------------------------------------------

### 2. Pose de la mire dans le LiDAR

-   Chargement du nuage `.asc`
-   Extraction du plan via RANSAC
-   Projection des points dans le plan
-   Approximation par rectangle minimal (`minAreaRect`)
-   Alignement par méthode d'Umeyama
-   Obtention de T_lidar_to_board

------------------------------------------------------------------------

### 3. Transformation finale

La transformation extrinsèque est calculée par :

    T_lidar_to_camera = T_board_to_camera @ inverse(T_lidar_to_board)

Une moyenne robuste est ensuite calculée sur toutes les paires valides.

------------------------------------------------------------------------

## Paramètres principaux

-   PLANE_DIST_THRESH : seuil RANSAC (m)
-   PLANE_ITERS : nombre d'itérations RANSAC
-   REPROJ_MAX_POINTS : sous-échantillonnage projection
-   REPROJ_MIN_Z : profondeur minimale (m)

------------------------------------------------------------------------

## Sorties générées

### 1. Extrinsèques moyennes

-   extrinsics_lidar_to_camera.json
-   extrinsics_lidar_to_camera.yaml

Contiennent : - Matrice 4×4 complète - Rotation 3×3 - Translation 3×1

------------------------------------------------------------------------

### 2. Extrinsèques par paire (optionnel)

Fichier généré pour la paire BEST_ID.

Toutes les paires ne donne pas de bons résultats. 
Garder les paramètres extrinsèques de celle dont la projection est la plus précise (dossier reprojection) 
et dont les paramètres de translation (dernière colone de la matrice des paramètres extrinsèques) 
sont les plus cohérents avec la disposition réel entre le lidar et la caméra. 
Pour le précedent datatset, c'était la paire 4.

------------------------------------------------------------------------

### 3. Visualisation reprojection

Dossier :

    dataset_test/reprojection/

Pour chaque paire : - XXXX_overlay_pair.jpg - XXXX_overlay_mean.jpg

------------------------------------------------------------------------

## Interprétation des résultats

La translation est exprimée en mètres.

La norme \|\|t\|\| représente la distance entre les centres LiDAR et
caméra.

Une bonne calibration doit produire : - Une projection cohérente
visuellement - Un alignement précis du damier dans l'image

Toutes les paires ne donne pas des projections cohérentes. 

------------------------------------------------------------------------

## Exécution

Depuis le dossier contenant le script :

    python3 calib_lidar_camera.py

------------------------------------------------------------------------

## Hypothèses

-   Le panneau est plan
-   Le damier est entièrement visible
-   Les fichiers image / nuage sont correctement synchronisés
