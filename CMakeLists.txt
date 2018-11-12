cmake_minimum_required(VERSION 2.8.3)
project(mono-explorer)
find_package(catkin REQUIRED
  COMPONENTS
  cv_bridge
  image_transport
  pcl_ros
  roscpp
  sensor_msgs
  tf2_ros
  )

include(ExternalProject)

ExternalProject_Add(dso
  GIT_REPOSITORY https://github.com/JakobEngel/dso.git
  INSTALL_COMMAND ""
  )

set(DSO_PREFIX ${CMAKE_BINARY_DIR}/dso_node/dso-prefix/src)
set(DSO_PATH ${DSO_PREFIX}/dso)
find_library(Dso_LIBS dso ${DSO_PREFIX}/dso-build/lib)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  )

add_executable(dso_node src/DsoNode.cpp)
target_link_libraries(dso_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Dso_LIBS})
