#include "detection_center/detection_center.hpp"
#include <fstream>
#include <iostream>

using namespace nvinfer1;


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

// Old TensorRT (<=8.x) binding-based API:
// static size_t getBindingSize(ICudaEngine* engine, int index) {
//     Dims dims = engine->getBindingDimensions(index);
//     size_t size = 1;
//     for (int i = 0; i < dims.nbDims; i++)
//         size *= dims.d[i];
//     return size;
// }

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
    // Old TensorRT (<=8.x):
    // inputSize_  = getBindingSize(engine_, 0) * sizeof(uint16_t);
    // outputSize_ = getBindingSize(engine_, 1) * sizeof(uint16_t);
    // New TensorRT (10+):
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

    // Old TensorRT (<=8.x) binding-based approach:
    // bindings_[0] = d_input_;
    // bindings_[1] = d_output_;
    // New TensorRT (10+) name-based approach:
    context_->setTensorAddress(inputName_.c_str(),  d_input_);
    context_->setTensorAddress(outputName_.c_str(), d_output_);

    // Cache input dimensions for preprocessing
    // Old TensorRT (<=8.x):
    // Dims dims   = engine_->getBindingDimensions(0);
    // New TensorRT (10+):
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

    // 3. Run inference
    cudaMemcpyAsync(d_input_, h_input_.data(), inputSize_, cudaMemcpyHostToDevice, stream_);
    // Old TensorRT (<=8.x):
    // context_->executeV2(bindings_);
    // New TensorRT (10+):
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(h_output_.data(), d_output_, outputSize_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

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

    // Filter rows by confidence threshold in one shot
    cv::Mat mask = maxScores > 0.5f;

    // Process surviving rows
    for (int i = 0; i < numAnchors_; i++) {
        if (!mask.at<uint8_t>(i)) continue; // skip in one check

        float* row = table.ptr<float>(i);

        // classId — find index of max score
        int classId = std::max_element(row + 4, row + numFields_) - (row + 4);

        float w  = row[2] * frame.cols;
        float h  = row[3] * frame.rows;
        float x1 = row[0] * frame.cols - w / 2.0f;  // top-left x
        float y1 = row[1] * frame.rows - h / 2.0f;  // top-left y

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

    return detections;
}