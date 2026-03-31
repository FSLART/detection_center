#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cstdint>

class DetectionCenter {
public:
    DetectionCenter(const std::string& enginePath);
    ~DetectionCenter();
    void detect(const cv::Mat& frame);

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* d_input_;
    void* d_output_;
    // Old TensorRT (<=8.x) binding-based API:
    // void* bindings_[2];
    // New TensorRT (10+) name-based API:
    cudaStream_t stream_;
    const char* inputName_;
    const char* outputName_;
    size_t inputSize_;
    size_t outputSize_;
    int inputHeight_;
    int inputWidth_;
    std::vector<uint16_t> h_input_;
    std::vector<uint16_t> h_output_;
    cv::Mat resized;
    std::vector<cv::Mat> channels;
};