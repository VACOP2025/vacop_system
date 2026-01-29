cmake_minimum_required(VERSION 3.5)
project(vacop_mapping)

# On trouve les outils
find_package(ament_cmake REQUIRED)

# On installe le dossier launch pour qu'il soit visible par ROS
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
