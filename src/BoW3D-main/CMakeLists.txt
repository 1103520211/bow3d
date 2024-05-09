cmake_minimum_required(VERSION 2.8.3)
project(bow3d)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_CXX_STANDARD 14)
find_package(catkin REQUIRED COMPONENTS  
  pcl_ros
  pcl_conversions
  roscpp
  roslib
  rospy  
  cv_bridge
  std_msgs  
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED QUIET)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}  
  ${CHOLMOD_INCLUDE_DIR}
  ${GTSAM_LIBRARY_DIRS}
)
    
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS pcl_ros roscpp roslib rospy std_msgs 
    DEPENDS EIGEN3 PCL GTSAM
)

add_executable(bow3d 
      src/Frame.cpp       
      src/LinK3D_Extractor.cpp 
      src/BoW3D.cpp 
      Example.cpp  
      )
target_link_libraries(bow3d ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${G2O_LIBS} ${Sophus_LIBRARIES})

add_executable(sc
      sc_text.cpp
      src/Scancontext.cpp
)
target_link_libraries(sc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})