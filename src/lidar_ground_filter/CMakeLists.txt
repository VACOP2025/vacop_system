cmake_minimum_required(VERSION 3.8)
project(lidar_ground_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ransac_ground_filter_node src/ransac_ground_filter_node.cpp)

ament_target_dependencies(ransac_ground_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
  tf2_ros
  tf2_sensor_msgs
  tf2_eigen
  Eigen3
)

target_link_libraries(ransac_ground_filter_node ${PCL_LIBRARIES} Eigen3::Eigen)

add_executable(euclidean_cluster_node src/euclidean_cluster_node.cpp)

ament_target_dependencies(euclidean_cluster_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  Eigen3
)

target_link_libraries(euclidean_cluster_node ${PCL_LIBRARIES} Eigen3::Eigen)

install(TARGETS 
  ransac_ground_filter_node
  euclidean_cluster_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()