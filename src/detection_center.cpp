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

    // Allocate buffers based on engine shape
    // d_input and d_output are the memory of the device, aka the GPU, hence why we need to use cudaMalloc
    // h_input and h_output are the memory of the host, aka the system's RAM
    // Old TensorRT (<=8.x):
    // inputSize_  = getBindingSize(engine_, 0) * sizeof(uint16_t);
    // outputSize_ = getBindingSize(engine_, 1) * sizeof(uint16_t);
    // New TensorRT (10+):
    inputSize_  = getTensorBindingSize(engine_, inputName_)  * sizeof(uint16_t);
    outputSize_ = getTensorBindingSize(engine_, outputName_) * sizeof(uint16_t);

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
    context_->setTensorAddress(inputName_,  d_input_);
    context_->setTensorAddress(outputName_, d_output_);

    // Cache input dimensions for preprocessing
    // Old TensorRT (<=8.x):
    // Dims dims   = engine_->getBindingDimensions(0);
    // New TensorRT (10+):
    Dims dims   = engine_->getTensorShape(inputName_);
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

void DetectionCenter::detect(const cv::Mat& frame) {
    // 1. Preprocess: resize and normalize
    cv::resize(frame, resized, cv::Size(inputWidth_, inputHeight_));
    resized.convertTo(resized, CV_16F, 1.0 / 255.0);

    // 2. Convert HWC → CHW
    cv::split(resized, channels);
    int channelSize = inputHeight_ * inputWidth_;
    for (int c = 0; c < 3; c++)
        memcpy(h_input_.data() + c * channelSize, channels[c].data, channelSize * sizeof(uint16_t));

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

    // Process output_fp32 here (decode boxes, NMS, publish results...)
}