#ifndef ZED_CENTER_H_
#define ZED_CENTER_H_

#include <rclcpp/rclcpp.hpp>
#include <sl/Camera.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <cmath>
#include <algorithm>
#include <array>
#include "rclcpp_components/register_node_macro.hpp"
#include <lart_msgs/msg/state.hpp>
#include <lart_msgs/srv/heartbeat.hpp>
#include <lart_msgs/msg/cone_array.hpp>
#include <lart_msgs/msg/cone.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include "detection_center/detection_center.hpp"

// includes for latency measure
#include <chrono>
#include <numeric>

#define RIGHT_IMG_FRAME_ID "zed_camera_right"
#define LEFT_IMG_FRAME_ID "zed_camera_left"
#define CAMERA_FRAME_ID "zed_camera_center"

using namespace sl;

class ZedCenter : public rclcpp::Node
{

public:
    explicit ZedCenter(const rclcpp::NodeOptions &options, DetectionCenter& detector);

private:
    Camera zed;
    RuntimeParameters runtime_parameters;
    int64_t frame_counter;
    std::chrono::steady_clock::time_point last_image_time;
    rclcpp::TimerBase::SharedPtr timer_;
    bool first_image = false;

    sl::CameraInformation cached_camera_info;
    sl::CalibrationParameters cached_calibration_params;
    
    rclcpp::Publisher<lart_msgs::msg::State>::SharedPtr emergency_pub;

    // Camera info template
    sensor_msgs::msg::CameraInfo left_camera_info_template;

    void setupCameraInfoTemplates();

    image_transport::Publisher left_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub;

    // Homography matrix (row-major 3x3, loaded from ROS parameter)
    std::array<double, 9> homography_matrix_;

    // for latency measure
    // std::vector<long long> latencies;

    rclcpp::Time last_capture_time;
    rclcpp::Service<lart_msgs::srv::Heartbeat>::SharedPtr timestamp_service_;
    rclcpp::Publisher<lart_msgs::msg::ConeArray>::SharedPtr cone_array_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub;
    rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_pub_;
    
    // Camera → base_footprint transform matrix (same as zed_bridge, using double)
    double transform_matrix_[4][4];

    // Marker IDs from previous frame (used to delete old markers)
    std::vector<int> marker_ids_;

    void publishImages();
    void handle_timestamp_request(const std::shared_ptr<lart_msgs::srv::Heartbeat::Request> request, std::shared_ptr<lart_msgs::srv::Heartbeat::Response> response);

    static int getOCVtype(sl::MAT_TYPE type);
    static cv::Mat slMat2cvMat(sl::Mat &input);

    DetectionCenter& detector_;
};

#endif // ZED_CENTER_H_
