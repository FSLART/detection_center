#include "detection_center/detection_center.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/cudawarping.hpp>

using namespace nvinfer1;


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

// New TensorRT (10+) name-based API:
static size_t getTensorBindingSize(ICudaEngine* engine, const char* name) {
    Dims dims = engine->getTensorShape(name);
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}

DetectionCenter::DetectionCenter(const std::string& enginePath) {
    // Load engine from disk
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }

    std::vector<char> engineData(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (engineData.empty()) {
        throw std::runtime_error("Engine file is empty: " + enginePath);
    }

    // Deserialize and create context
    runtime_ = createInferRuntime(logger);
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_) {
        delete runtime_;
        throw std::runtime_error("Failed to deserialize engine (file may be corrupt or TensorRT version mismatch)");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        delete engine_;
        delete runtime_;
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // Clear the now unnecessary engineData from memory
    engineData.clear();
    engineData.shrink_to_fit();

    // Resolve tensor names (TensorRT 10+ name-based API)
    inputName_  = engine_->getIOTensorName(0);
    outputName_ = engine_->getIOTensorName(1);

    Dims outDims = engine_->getTensorShape(outputName_.c_str()); // [batch_size, num_fields, num_anchors]
    numFields_  = outDims.d[1]; // 9 (4 coords + 5 classes)
    numAnchors_ = outDims.d[2];

    // Allocate buffers based on engine shape
    // d_input and d_output are the memory of the device, aka the GPU, hence why we need to use cudaMalloc
    // h_input and h_output are the memory of the host, aka the system's RAM
    inputSize_  = getTensorBindingSize(engine_, inputName_.c_str())  * sizeof(uint16_t);
    outputSize_ = getTensorBindingSize(engine_, outputName_.c_str()) * sizeof(uint16_t);

    cudaError_t err;
    err = cudaMalloc(&d_input_, inputSize_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc d_input_ failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_output_, outputSize_);
    if (err != cudaSuccess) {
        cudaFree(d_input_);
        throw std::runtime_error(std::string("cudaMalloc d_output_ failed: ") + cudaGetErrorString(err));
    }

    // Create CUDA stream for async inference
    cudaStreamCreate(&stream_);

    // Here we need to use uint16_t because our model runs on FP16 precision
    h_input_.resize(inputSize_  / sizeof(uint16_t));
    h_output_.resize(outputSize_ / sizeof(uint16_t));

    // New TensorRT (10+) name-based approach:
    context_->setTensorAddress(inputName_.c_str(),  d_input_);
    context_->setTensorAddress(outputName_.c_str(), d_output_);

    // Cache input dimensions for preprocessing
    Dims dims   = engine_->getTensorShape(inputName_.c_str());
    inputHeight_ = dims.d[2];
    inputWidth_  = dims.d[3];
}

DetectionCenter::~DetectionCenter() {
    cudaStreamDestroy(stream_);
    cudaFree(d_input_);
    cudaFree(d_output_);
    delete context_;
    delete engine_;
    delete runtime_;
}

std::vector<DetectionCenter::Detection> DetectionCenter::detect(const cv::Mat& frame) {

    // clear detection results from previous frames
    boxes_.clear();
    scores_.clear();
    classIds_.clear();

    // 1. Preprocess: resize and normalize
    cv::resize(frame, resized_, cv::Size(inputWidth_, inputHeight_));
    resized_.convertTo(resized_, CV_16F, 1.0 / 255.0);

    // 2. Convert HWC → CHW
    cv::split(resized_, channels_);
    int channelSize = inputHeight_ * inputWidth_;
    for (int c = 0; c < 3; c++)
        memcpy(h_input_.data() + c * channelSize, channels_[c].data, channelSize * sizeof(uint16_t));
    
    auto start_time_infer = std::chrono::high_resolution_clock::now();

    // 3. Run inference
    cudaError_t err = cudaMemcpyAsync(d_input_, h_input_.data(), inputSize_, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaMemcpyAsync (Host->Device) failed: " << cudaGetErrorString(err) << std::endl;
    }

    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[DetectionCenter] ERROR: TensorRT enqueueV3 failed!" << std::endl;
    }

    err = cudaMemcpyAsync(h_output_.data(), d_output_, outputSize_, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaMemcpyAsync (Device->Host) failed: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
    }
    auto end_time_infer = std::chrono::high_resolution_clock::now();
    auto duration_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_infer - start_time_infer);
    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Inference Time: %ld ms", duration_infer.count() );
    
    // 4. Convert output back to float32 for processing
    cv::Mat output_fp16(1, h_output_.size(), CV_16F, h_output_.data());
    cv::Mat output_fp32;
    output_fp16.convertTo(output_fp32, CV_32F);

    // ------------------------ Decode boxes, NMS ------------------------

    // Reshape and transpose as before
    cv::Mat reshaped(numFields_, numAnchors_, CV_32F, output_fp32.data);
    cv::Mat table;
    cv::transpose(reshaped, table);

    // Extract all class scores at once (columns 4-8) and find max per row
    cv::Mat classScores = table.colRange(4, numFields_);
    cv::Mat maxScores;
    cv::reduce(classScores, maxScores,  1, cv::REDUCE_MAX);  // max score per row

    double minScore, maxScore;
    cv::minMaxLoc(maxScores, &minScore, &maxScore);

    // Filter rows by confidence threshold in one shot
    cv::Mat mask = maxScores > 0.5f;

    // Process surviving rows
    for (int i = 0; i < numAnchors_; i++) {
        if (!mask.at<uint8_t>(i)) continue; // skip in one check

        float* row = table.ptr<float>(i);

        // classId — find index of max score
        int classId = std::max_element(row + 4, row + numFields_) - (row + 4);

        float rx = static_cast<float>(frame.cols) / inputWidth_;
        float ry = static_cast<float>(frame.rows) / inputHeight_;
        float w  = row[2] * rx;
        float h  = row[3] * ry;
        float x1 = row[0] * rx - w / 2.0f;  // top-left x
        float y1 = row[1] * ry - h / 2.0f;  // top-left y

        boxes_.emplace_back(x1, y1, w, h);
        scores_.emplace_back(maxScores.at<float>(i));
        classIds_.emplace_back(classId);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_, scores_, 0.5f, 0.45f, indices);

    // Build result vector with only NMS survivors
    std::vector<Detection> detections;
    detections.reserve(indices.size());

    for (int idx : indices) {
        Detection det;
        det.box = boxes_[idx];
        det.score = scores_[idx];
        det.classId = classIds_[idx];
        detections.push_back(det);
    }

    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "BBoxes Found: %i", indices.size() );
    
    return detections;
}

std::vector<DetectionCenter::Detection> DetectionCenter::detect(const cv::cuda::GpuMat& gpu_frame) {
    
    auto start_time_detect = std::chrono::high_resolution_clock::now();

    // clear detection results from previous frames
    boxes_.clear();
    scores_.clear();
    classIds_.clear();

    // 1. Resize on GPU (fast: avoids resizing full 1920x1200 on CPU)
    cv::cuda::resize(gpu_frame, gpu_resized_, cv::Size(inputWidth_, inputHeight_));

    // 2. Download the small resized image to CPU, then use the proven CPU preprocessing
    //    from commit 3d51f35 (the "its working" commit). The image is now only 640x640,
    //    so this download is very cheap (~1.2MB vs ~6.9MB for the full-res image).
    gpu_resized_.download(resized_);
    resized_.convertTo(resized_, CV_16F, 1.0 / 255.0);

    // 3. Convert HWC → CHW (same proven logic as the CPU detect() path)
    cv::split(resized_, channels_);
    int channelSize = inputHeight_ * inputWidth_;
    for (int c = 0; c < 3; c++)
        memcpy(h_input_.data() + c * channelSize, channels_[c].data, channelSize * sizeof(uint16_t));

    // 4. Upload prepared FP16 CHW tensor to GPU
    cudaError_t upload_err = cudaMemcpyAsync(d_input_, h_input_.data(), inputSize_, cudaMemcpyHostToDevice, stream_);
    if (upload_err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaMemcpyAsync (Host->Device) failed: " << cudaGetErrorString(upload_err) << std::endl;
    }

    // --- OLD GPU preprocessing (broken: produces corrupted FP16 data, zero detections) ---
    // cv::cuda::GpuMat gpu_fp32;
    // gpu_resized_.convertTo(gpu_fp32, CV_32FC3, 1.0 / 255.0);
    // std::vector<cv::cuda::GpuMat> f32_channels(3);
    // cv::cuda::split(gpu_fp32, f32_channels);
    // int channelSize = inputHeight_ * inputWidth_;
    // std::vector<cv::cuda::GpuMat> f16_channels(3);
    // for (int c = 0; c < 3; c++) {
    //     f32_channels[c].convertTo(f16_channels[c], CV_16F);
    //     cudaMemcpy2DAsync(
    //         static_cast<uint16_t*>(d_input_) + c * channelSize,
    //         inputWidth_ * sizeof(uint16_t),
    //         f16_channels[c].data, f16_channels[c].step,
    //         inputWidth_ * sizeof(uint16_t), inputHeight_,
    //         cudaMemcpyDeviceToDevice, 0);
    // }
    // cudaStreamSynchronize(0);
    
    auto start_time_infer = std::chrono::high_resolution_clock::now();

    // 5. Run inference
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[DetectionCenter] ERROR: TensorRT enqueueV3 failed!" << std::endl;
    }

    // 6. Download FP16 output to CPU and convert there (safe, proven, fast for ~150KB)
    cudaError_t err = cudaMemcpyAsync(h_output_.data(), d_output_, outputSize_,
                                       cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaMemcpyAsync (Device->Host) failed: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        std::cerr << "[DetectionCenter] ERROR: cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
    }
    auto end_time_infer = std::chrono::high_resolution_clock::now();
    auto duration_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_infer - start_time_infer);
    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Inference Time (GPU Path): %ld ms", duration_infer.count() );

    // 7. Convert output FP16 → FP32
    // --- Option A: Native ARM hardware cast (potentially faster, bypasses OpenCV emulation) ---
    cv::Mat output_fp32(1, h_output_.size(), CV_32F);
    __fp16* raw_fp16 = reinterpret_cast<__fp16*>(h_output_.data());
    float* out_fp32_ptr = output_fp32.ptr<float>();
    int total_elements = h_output_.size();
    for (int i = 0; i < total_elements; ++i) {
        out_fp32_ptr[i] = static_cast<float>(raw_fp16[i]);
    }
    // --- Option B: Original OpenCV path (uncomment if Option A causes issues) ---
    // cv::Mat output_fp16(1, h_output_.size(), CV_16F, h_output_.data());
    // cv::Mat output_fp32;
    // output_fp16.convertTo(output_fp32, CV_32F);

    // ------------------------ Decode boxes, NMS ------------------------

    // Reshape and transpose as before
    cv::Mat reshaped(numFields_, numAnchors_, CV_32F, output_fp32.data);
    cv::Mat table;
    cv::transpose(reshaped, table);

    // Extract all class scores at once (columns 4-8) and find max per row
    cv::Mat classScores = table.colRange(4, numFields_);
    cv::Mat maxScores;
    cv::reduce(classScores, maxScores,  1, cv::REDUCE_MAX);  // max score per row

    double minScore, maxScore;
    cv::minMaxLoc(maxScores, &minScore, &maxScore);

    // Filter rows by confidence threshold in one shot
    cv::Mat mask = maxScores > 0.5f;

    // Process surviving rows
    for (int i = 0; i < numAnchors_; i++) {
        if (!mask.at<uint8_t>(i)) continue; // skip in one check

        float* row = table.ptr<float>(i);

        // classId — find index of max score
        int classId = std::max_element(row + 4, row + numFields_) - (row + 4);

        float rx = static_cast<float>(gpu_frame.cols) / inputWidth_;
        float ry = static_cast<float>(gpu_frame.rows) / inputHeight_;
        float w  = row[2] * rx;
        float h  = row[3] * ry;
        float x1 = row[0] * rx - w / 2.0f;  // top-left x
        float y1 = row[1] * ry - h / 2.0f;  // top-left y

        boxes_.emplace_back(x1, y1, w, h);
        scores_.emplace_back(maxScores.at<float>(i));
        classIds_.emplace_back(classId);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_, scores_, 0.5f, 0.45f, indices);

    // Build result vector with only NMS survivors
    std::vector<Detection> detections;
    detections.reserve(indices.size());

    for (int idx : indices) {
        Detection det;
        det.box = boxes_[idx];
        det.score = scores_[idx];
        det.classId = classIds_[idx];
        detections.push_back(det);
    }

    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "BBoxes Found (GPU Path): %li", indices.size() );

    auto end_time_detect = std::chrono::high_resolution_clock::now();
    auto duration_detect = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_detect - start_time_detect);
    RCLCPP_INFO(rclcpp::get_logger("detection_center"), "Detection Call Time: %ld ms", duration_detect.count() );

    return detections;
}