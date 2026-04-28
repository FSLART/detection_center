#include <rclcpp/rclcpp.hpp>
#include "detection_center/zed_center.hpp"
#include "detection_center/detection_center.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    DetectionCenter detector("/home/lart-tasha/Documents/repos/ros2_ws/src/mapper_speedrun/model/yolo8n1024.trt");
    auto zed_node = std::make_shared<ZedCenter>(rclcpp::NodeOptions(), detector);

    rclcpp::spin(zed_node);
    rclcpp::shutdown();
    return 0;
}