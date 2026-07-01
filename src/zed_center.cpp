#include "detection_center/zed_center.hpp"

ZedCenter::ZedCenter(const rclcpp::NodeOptions& options, DetectionCenter& detector) : Node("zed_center", options), detector_(detector)
{
    this->emergency_pub = this->create_publisher<lart_msgs::msg::State>("/pc_origin/emergency", 10);

    // set configuration parameters
    // https://www.stereolabs.com/docs/video/camera-controls
    InitParameters init_params;
    init_params.sdk_verbose = 1;
    init_params.camera_resolution = RESOLUTION::HD1200;
    init_params.camera_fps = 60;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::NONE;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params.enable_right_side_measure = true;

    // set runtime parameters
    this->runtime_parameters.enable_depth = false;

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
    this->left_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("/zed/left/camera_info", 10);

    this->cone_array_pub = this->create_publisher<lart_msgs::msg::ConeArray>("/mapping/cones", 10);
    this->marker_array_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mapping/cones_markers", 10);
    this->annotations_pub_ = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>("/zed/image_annotations", 10);

    // Load Homography matrix from ROS parameter (9 values, row-major 3x3)
    this->declare_parameter("homography_matrix", std::vector<double>(9, 0.0));
    auto h_vec = this->get_parameter("homography_matrix").as_double_array();
    for (size_t i = 0; i < 9; ++i) {
        this->homography_matrix_[i] = h_vec[i];
    }
    
    // Cache camera info once instead of getting it every frame
    this->cached_camera_info = zed.getCameraInformation();
    this->cached_calibration_params = cached_camera_info.camera_configuration.calibration_parameters;

    // Pre-populate camera info message templates
    setupCameraInfoTemplates();
    this->last_image_time = std::chrono::steady_clock::now();

    // base_footprint transform matrix
    transform_matrix_[0][0] = 1.0; transform_matrix_[0][1] = 0.0; transform_matrix_[0][2] = 0.0; transform_matrix_[0][3] = -0.5;
    transform_matrix_[1][0] = 0.0; transform_matrix_[1][1] = 1.0; transform_matrix_[1][2] = 0.0; transform_matrix_[1][3] = 0.0;
    transform_matrix_[2][0] = 0.0; transform_matrix_[2][1] = 0.0; transform_matrix_[2][2] = 1.0; transform_matrix_[2][3] = 0.95;
    transform_matrix_[3][0] = 0.0; transform_matrix_[3][1] = 0.0; transform_matrix_[3][2] = 0.0; transform_matrix_[3][3] = 1.0;

    // Create heartbeat service
    this->timestamp_service_ = this->create_service<lart_msgs::srv::Heartbeat>("zed/last_timestamp", std::bind(&ZedCenter::handle_timestamp_request, this, std::placeholders::_1, std::placeholders::_2));

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(16), 
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
}

void ZedCenter::publishImages()
{
    // Easy to adjust: set to 1 to publish every frame (disables skip entirely)
    static constexpr int IMAGE_PUBLISH_EVERY_N_FRAMES = 3;

    auto start_time_frame = std::chrono::high_resolution_clock::now();

    auto start_grab = std::chrono::high_resolution_clock::now();
    auto err = zed.grab(this->runtime_parameters);
    auto end_grab = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Grab Time: %ld ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_grab - start_grab).count());

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
        static lart_msgs::msg::ConeArray cone_array;
        static visualization_msgs::msg::MarkerArray marker_array;
        static foxglove_msgs::msg::ImageAnnotations annotations_msg;

        // Clear containers instead of recreating
        cone_array.cones.clear();
        marker_array.markers.clear();
        annotations_msg.points.clear();
        annotations_msg.texts.clear();

        // convert the image to OpenCV format
        left_image_cv_bgra = slMat2cvMat(left_image);
        cv::cvtColor(left_image_cv_bgra, left_image_cv_rgb, cv::COLOR_BGRA2RGB);

        // 2. Convert BGRA → RGB on GPU
        cv::cuda::cvtColor(gpu_left_bgra_, gpu_left_rgb_, cv::COLOR_BGRA2RGB);

        // 3. Only download and publish the image on selected frames (JPEG compress is expensive)
        if (this->frame_counter % IMAGE_PUBLISH_EVERY_N_FRAMES == 0) {
            gpu_left_rgb_.download(left_image_cv_rgb);

            // convert the image to a ROS message
            left_image_msg.header.stamp = timestamp;
            left_image_msg.header.frame_id = LEFT_IMG_FRAME_ID;
            left_image_msg.height = left_image_cv_rgb.rows;
            left_image_msg.width = left_image_cv_rgb.cols;
            left_image_msg.encoding = "rgb8";
            left_image_msg.step = left_image_cv_rgb.step;

            left_image_msg.data.assign(left_image_cv_rgb.data, left_image_cv_rgb.data + left_image_cv_rgb.rows * left_image_cv_rgb.cols * left_image_cv_rgb.channels());
            
            left_camera_info_template.header.stamp = timestamp;

            auto start_pub = std::chrono::high_resolution_clock::now();
            this->left_image_pub.publish(left_image_msg);
            auto end_pub = std::chrono::high_resolution_clock::now();
            RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Image Publish Time: %ld ms",
                std::chrono::duration_cast<std::chrono::milliseconds>(end_pub - start_pub).count());

            this->left_info_pub->publish(left_camera_info_template);
        }

        // --- Run YOLO inference (returns raw 2D detections) ---
        // Detection ALWAYS runs, regardless of whether we published the image
        auto detections = detector_.detect(gpu_left_rgb_);



        // Reserve space for detections
        cone_array.cones.reserve(detections.size());
        marker_array.markers.reserve(detections.size() + this->marker_ids_.size());

        // Delete old markers from previous frame
        for (const auto &marker_id : this->marker_ids_)
        {
            visualization_msgs::msg::Marker old_marker;
            old_marker.header.frame_id = "base_footprint";
            old_marker.header.stamp = timestamp;
            old_marker.ns = "cone_marker";
            old_marker.id = marker_id;
            old_marker.action = visualization_msgs::msg::Marker::DELETE;
            marker_array.markers.push_back(std::move(old_marker));
        }
        this->marker_ids_.clear();

        // --- Process each detection ---
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto &det = detections[i];
            const cv::Rect &box = det.box;

            // STEP 1: Bottom-center pixel (footpoint where cone touches the ground)
            double u = box.x + box.width / 2.0;
            double v = box.y + box.height;  // bottom edge, not center

            // STEP 2: Homography projection (pixel → ground plane in meters)
            const double* H = homography_matrix_.data();
            double W = H[6] * u + H[7] * v + H[8];
            if (std::abs(W) < 1e-3) continue;  // reject near-horizon points

            double X_ground = (H[0] * u + H[1] * v + H[2]) / W;  // forward (meters)
            double Y_ground = (H[3] * u + H[4] * v + H[5]) / W;  // right (meters)

            // STEP 3: Map to ZED coordinate convention (X=forward, Y=left, Z=up)
            double obj_x =  X_ground;   // forward
            double obj_y = -Y_ground;   // left (invert right)
            double obj_z =  0.0;        // cone is on the ground plane

            // Distance filter (squared to avoid sqrt, same as zed_bridge.cpp)
            double distance_sq = obj_x * obj_x + obj_y * obj_y;

            // STEP 5: Apply transform matrix (camera → base_footprint)
            double transformed_x = transform_matrix_[0][0] * obj_x + transform_matrix_[0][1] * obj_y +
                                    transform_matrix_[0][2] * obj_z + transform_matrix_[0][3];
            double transformed_y = transform_matrix_[1][0] * obj_x + transform_matrix_[1][1] * obj_y +
                                    transform_matrix_[1][2] * obj_z + transform_matrix_[1][3];

            // --- Create Cone message ---
            if (distance_sq >= 0.25 && distance_sq <= 650.0)  // 0.5m to ~25.5m
            {
                lart_msgs::msg::Cone cone;
                cone.header.frame_id = "base_footprint";
                cone.position.x = transformed_x;
                cone.position.y = transformed_y;
                cone.position.z = 0.0;
                cone.class_type.data = det.classId;
                cone_array.cones.push_back(std::move(cone));
            }

            // --- Create Marker ---
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "base_footprint";
            marker.header.stamp = timestamp;
            marker.ns = "cone_marker";
            marker.id = this->frame_counter * 1000 + i;
            this->marker_ids_.push_back(marker.id);

            marker.type = visualization_msgs::msg::Marker::CYLINDER;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.lifetime = rclcpp::Duration::from_seconds(0.2);

            marker.pose.position.x = transformed_x;
            marker.pose.position.y = transformed_y;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.23;
            marker.scale.y = 0.23;
            marker.scale.z = 0.31;

            // --- Prepare Foxglove 2D annotation ---
            foxglove_msgs::msg::PointsAnnotation poly;
            poly.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;
            poly.thickness = 3.0;

            // Color marker and annotation based on class (same as zed_bridge.cpp)
            switch (det.classId)
            {
            case 1: // Yellow
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
                poly.outline_color.r = 1.0; poly.outline_color.g = 1.0; poly.outline_color.b = 0.0; poly.outline_color.a = 1.0;
                break;
            case 2: // Blue
                marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 1.0;
                poly.outline_color.r = 0.0; poly.outline_color.g = 0.0; poly.outline_color.b = 1.0; poly.outline_color.a = 1.0;
                break;
            case 3: // Lil Orange
                marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 1.0;
                poly.outline_color.r = 1.0; poly.outline_color.g = 0.5; poly.outline_color.b = 0.0; poly.outline_color.a = 1.0;
                break;
            case 4: // Big Orange
                marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 1.0;
                marker.scale.x = 0.35; marker.scale.y = 0.35; marker.scale.z = 0.50;
                poly.outline_color.r = 1.0; poly.outline_color.g = 0.5; poly.outline_color.b = 0.0; poly.outline_color.a = 1.0;
                break;
            default: // Default (Yellow)
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
                poly.outline_color.r = 1.0; poly.outline_color.g = 1.0; poly.outline_color.b = 0.0; poly.outline_color.a = 1.0;
            }

            marker_array.markers.push_back(std::move(marker));

            // Build the 2D bounding box polygon from the 4 corners of the cv::Rect
            foxglove_msgs::msg::Point2 p1, p2, p3, p4;
            p1.x = box.x;               p1.y = box.y;                // Top-Left
            p2.x = box.x + box.width;   p2.y = box.y;                // Top-Right
            p3.x = box.x + box.width;   p3.y = box.y + box.height;   // Bottom-Right
            p4.x = box.x;               p4.y = box.y + box.height;   // Bottom-Left

            poly.points.push_back(p1);
            poly.points.push_back(p2);
            poly.points.push_back(p3);
            poly.points.push_back(p4);
            annotations_msg.points.push_back(std::move(poly));

            // Add confidence text above the bounding box
            foxglove_msgs::msg::TextAnnotation txt;
            txt.position.x = box.x;
            txt.position.y = box.y - 15;  // Slightly above the top-left corner
            txt.text = std::to_string(static_cast<int>(det.score * 100)) + "%";
            txt.font_size = 20.0;
            txt.text_color.r = 1.0; txt.text_color.g = 1.0; txt.text_color.b = 1.0; txt.text_color.a = 1.0;
            annotations_msg.texts.push_back(std::move(txt));
        }

        // Publish cone array, markers, and annotations
        this->cone_array_pub->publish(std::move(cone_array));
        this->marker_array_pub->publish(std::move(marker_array));
        this->annotations_pub_->publish(annotations_msg);

        this->frame_counter++;
        this->first_image = true;
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
    auto end_time_frame = std::chrono::high_resolution_clock::now();
    auto duration_frame = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_frame - start_time_frame);
    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Frame Processing Time: %ld ms", duration_frame.count() );
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

void ZedCenter::handle_timestamp_request(
    const std::shared_ptr<lart_msgs::srv::Heartbeat::Request> request,
    std::shared_ptr<lart_msgs::srv::Heartbeat::Response> response)
{
    (void)request;

    response->timestamp = this->last_capture_time;
}