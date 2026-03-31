#include <rclcpp/rclcpp.hpp>
#include "detection_center/zed_center.hpp"
#include "detection_center/detection_center.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: detection_center <path_to_engine>" << std::endl;
        return 1;
    }

    DetectionCenter detector(argv[1]);
    auto zed_node = std::make_shared<ZedCenter>(rclcpp::NodeOptions(), detector);

    rclcpp::spin(zed_node);
    rclcpp::shutdown();
    return 0;
}