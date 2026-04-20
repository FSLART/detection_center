#include "detection_center/zed_center.hpp"

ZedCenter::ZedCenter(const rclcpp::NodeOptions& options, DetectionCenter& detector) : Node("zed_center", options), detector_(detector)
{
    this->emergency_pub = this->create_publisher<lart_msgs::msg::State>("/pc_origin/emergency", 10);

    // set configuration parameters
    // https://www.stereolabs.com/docs/video/camera-controls
    InitParameters init_params;
    init_params.sdk_verbose = 1;
    init_params.camera_resolution = RESOLUTION::HD1200;
    init_params.depth_minimum_distance = 0.5;
    init_params.depth_maximum_distance = 25.0;
    init_params.camera_fps = 30;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::NEURAL_PLUS; // previous: PERFORMANCE, ULTRA, NEURAL_PLUS
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params.enable_right_side_measure = true;
    init_params.depth_stabilization = true;

    // set runtime parameters
    this->runtime_parameters.enable_depth = true;
    this->runtime_parameters.enable_fill_mode = false;
    this->runtime_parameters.confidence_threshold = 70;

    // open the camera
    auto err = this->zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS)
    {
        RCLCPP_WARN(this->get_logger(), "FAILURE: %d %d", (int)ERROR_CODE::CAMERA_NOT_DETECTED, (int)err);
        RCLCPP_ERROR(this->get_logger(), "Failed to open ZED camera");
        lart_msgs::msg::State emergency;
        emergency.data = lart_msgs::msg::State::EMERGENCY;
        this->emergency_pub->publish(emergency);
        throw std::runtime_error("Failed to open ZED camera");
    }

    // Define the ROI rectangle for AEC
    sl::Rect roi;
    roi.x = 960;
    roi.y = 600;
    roi.width = 1030;
    roi.height = 400;

    // Apply ROI for AEC/AGC
    this->zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC, roi, SIDE::BOTH, true);

    PositionalTrackingParameters tracking_params;
    this->zed.enablePositionalTracking(tracking_params);


    this->frame_counter = 0;

    this->left_image_pub = image_transport::create_publisher(this, "/zed/left/image_raw");
    this->depth_image_pub = image_transport::create_publisher(this, "/zed/depth/image_raw");

    this->left_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("/zed/left/camera_info", 10);
    this->depth_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("/zed/depth/camera_info", 10);
    
    // Cache camera info once instead of getting it every frame
    this->cached_camera_info = zed.getCameraInformation();
    this->cached_calibration_params = cached_camera_info.camera_configuration.calibration_parameters;

    // Pre-populate camera info message templates
    setupCameraInfoTemplates();
    this->last_image_time = std::chrono::steady_clock::now();

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(30), 
        std::bind(&ZedCenter::publishImages, this)
    );
}

void ZedCenter::setupCameraInfoTemplates()
{
    // Set up left camera info template
    left_camera_info_template.distortion_model = "plumb_bob";
    left_camera_info_template.d.resize(5);
    left_camera_info_template.d[0] = cached_calibration_params.left_cam.disto[0];
    left_camera_info_template.d[1] = cached_calibration_params.left_cam.disto[1];
    left_camera_info_template.d[2] = cached_calibration_params.left_cam.disto[2];
    left_camera_info_template.d[3] = cached_calibration_params.left_cam.disto[3];
    left_camera_info_template.d[4] = cached_calibration_params.left_cam.disto[4];
    left_camera_info_template.k.fill(0.0);
    left_camera_info_template.k[0] = cached_calibration_params.left_cam.fx;
    left_camera_info_template.k[2] = cached_calibration_params.left_cam.cx;
    left_camera_info_template.k[4] = cached_calibration_params.left_cam.fy;
    left_camera_info_template.k[5] = cached_calibration_params.left_cam.cy;
    left_camera_info_template.k[8] = 1.0;
    left_camera_info_template.p.fill(0.0);
    left_camera_info_template.p[0] = cached_calibration_params.left_cam.fx;
    left_camera_info_template.p[2] = cached_calibration_params.left_cam.cx;
    left_camera_info_template.p[5] = cached_calibration_params.left_cam.fy;
    left_camera_info_template.p[6] = cached_calibration_params.left_cam.cy;
    left_camera_info_template.p[10] = 1.0;

    // Copy for depth camera info template
    depth_camera_info_template = left_camera_info_template;
}

void ZedCenter::publishImages()
{
    auto err = zed.grab(this->runtime_parameters);

    if (err == ERROR_CODE::SUCCESS)
    {
        // Cache timestamp once
        const auto timestamp = this->now();
        this->last_capture_time = timestamp;
        this->last_image_time = std::chrono::steady_clock::now();

        // retrieve the left image
        sl::Mat left_image;
        zed.retrieveImage(left_image, VIEW::LEFT);

        // Pre-allocate reusable objects as static to avoid repeated allocations
        static cv::Mat left_image_cv_bgra;
        static cv::Mat left_image_cv_rgb;
        static sensor_msgs::msg::Image left_image_msg;
        static sensor_msgs::msg::Image depth_image_msg;

        // convert the image to OpenCV format
        left_image_cv_bgra = slMat2cvMat(left_image);
        cv::cvtColor(left_image_cv_bgra, left_image_cv_rgb, cv::COLOR_BGRA2RGB);

        // convert the image to a ROS message
        left_image_msg.header.stamp = timestamp;
        left_image_msg.header.frame_id = LEFT_IMG_FRAME_ID;
        left_image_msg.height = left_image_cv_rgb.rows;
        left_image_msg.width = left_image_cv_rgb.cols;
        left_image_msg.encoding = "rgb8";
        left_image_msg.step = left_image_cv_rgb.step;

        left_image_msg.data.assign(left_image_cv_rgb.data, left_image_cv_rgb.data + left_image_cv_rgb.rows * left_image_cv_rgb.cols * left_image_cv_rgb.channels());
        
        left_camera_info_template.header.stamp = timestamp;
        this->left_image_pub.publish(left_image_msg);
        this->left_info_pub->publish(left_camera_info_template);

        // Retrieve depth map
        sl::Mat depth_map;
        zed.retrieveMeasure(depth_map, sl::MEASURE::DEPTH);
        cv::Mat depth_cv = cv::Mat(depth_map.getHeight(), depth_map.getWidth(), 
                                   CV_32FC1, depth_map.getPtr<sl::uchar1>(MEM::CPU),
                                   depth_map.getStepBytes(MEM::CPU));
        // Build intrinsics from cached calibration
        CameraIntrinsics intrinsics;
        intrinsics.fx = cached_calibration_params.left_cam.fx;
        intrinsics.fy = cached_calibration_params.left_cam.fy;
        intrinsics.cx = cached_calibration_params.left_cam.cx;
        intrinsics.cy = cached_calibration_params.left_cam.cy;

        detector_.detect(left_image_cv_rgb, depth_cv, intrinsics);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "ZED camera grab failed");
        if (std::chrono::steady_clock::now() - this->last_image_time > std::chrono::seconds(1))
        {
            RCLCPP_ERROR(this->get_logger(), "ZED camera not responding, publishing emergency state");
            lart_msgs::msg::State emergency;
            emergency.data = lart_msgs::msg::State::EMERGENCY;
            this->emergency_pub->publish(emergency);
        }
    }
}

// Mapping between MAT_TYPE and CV_TYPE
int ZedCenter::getOCVtype(sl::MAT_TYPE type)
{
    int cv_type = -1;
    switch (type)
    {
    case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
    default: break;
    }
    return cv_type;
}

/**
 * Conversion function between sl::Mat and cv::Mat
 **/
cv::Mat ZedCenter::slMat2cvMat(sl::Mat &input)
{
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}