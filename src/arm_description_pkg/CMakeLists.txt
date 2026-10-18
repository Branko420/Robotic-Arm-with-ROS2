cmake_minimum_required(VERSION 3.8)
project(arm_description_pkg)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
