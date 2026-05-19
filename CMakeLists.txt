cmake_minimum_required(VERSION 3.8)
project(detection_center)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(lart_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_CUDA_VERSION AND NOT DEFINED OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV was NOT built with CUDA support! The zero-copy GPU preprocessing pipeline will fail to compile. Please rebuild OpenCV with WITH_CUDA=ON.")
endif()

# Find CUDA
# find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(CUDAToolkit REQUIRED)


# Find foxglove_msgs
find_package(foxglove_msgs REQUIRED)

# Find visualization_msgs
find_package(visualization_msgs REQUIRED)

# TensorRT is installed via apt on Jetson, so it is in standard system paths.
# I believe that we don't need to hardcode TENSORRT_ROOT, INCLUDE_DIRS, or LIB_DIRS.

# set(TENSORRT_ROOT /usr/local/tensorrt)
# set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
# set(TENSORRT_LIB_DIRS ${TENSORRT_ROOT}/lib)

# Find ZED SDK
find_package(ZED 3 REQUIRED)

# Define the executable and its source files
add_executable(detection_center
    src/main.cpp
    src/detection_center.cpp
    src/zed_center.cpp
)

# Include directories
target_include_directories(detection_center PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
    # ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(detection_center
    CUDA::cudart
    # ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    # -L${TENSORRT_LIB_DIRS}
    nvinfer
)

# Link ROS2 dependencies
ament_target_dependencies(detection_center
    rclcpp
    rclcpp_components
    sensor_msgs
    lart_msgs
    image_transport
    cv_bridge
    foxglove_msgs
    visualization_msgs
)

# Install the executable so ros2 run can find it
install(TARGETS detection_center
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()