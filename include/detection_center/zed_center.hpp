#ifndef ZED_CENTER_H_
#define ZED_CENTER_H_

#include <rclcpp/rclcpp.hpp>
#include <sl/Camera.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <cmath>
#include "rclcpp_components/register_node_macro.hpp"
#include <lart_msgs/msg/state.hpp>
#include <lart_msgs/srv/heartbeat.hpp>
#include <lart_msgs/msg/cone_array.hpp>
#include <lart_msgs/msg/cone.hpp>
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

    // Camera info templates
    sensor_msgs::msg::CameraInfo left_camera_info_template;
    sensor_msgs::msg::CameraInfo depth_camera_info_template;

    void setupCameraInfoTemplates();

    // std::shared_ptr<image_transport::ImageTransport> it; // Declare the ImageTransport object
    image_transport::Publisher left_image_pub;  // Declare the publisher for the left image
    image_transport::Publisher depth_image_pub; // Declare the publisher for the depth image

    // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub;
    // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub;

    // for latency measure
    // std::vector<long long> latencies;

    rclcpp::Time last_capture_time;
    rclcpp::Service<lart_msgs::srv::Heartbeat>::SharedPtr timestamp_service_;

    rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_pub_;
    
    void publishImages();

    static int getOCVtype(sl::MAT_TYPE type);
    static cv::Mat slMat2cvMat(sl::Mat &input);

    DetectionCenter& detector_;
};

#endif // ZED_CENTER_H_
