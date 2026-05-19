#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <chrono>

class DetectionCenter {
public:
    // Raw 2D detection result from YOLO inference
    struct Detection {
        cv::Rect box;      // bounding box in pixel coordinates
        float score;        // confidence score
        int classId;        // class index
    };

    DetectionCenter(const std::string& enginePath);
    ~DetectionCenter();

    std::vector<Detection> detect(const cv::Mat& frame);
    std::vector<Detection> detect(const cv::cuda::GpuMat& gpu_frame);

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
    std::string inputName_;
    std::string outputName_;
    size_t inputSize_;
    size_t outputSize_;
    int numFields_;
    int numAnchors_;

    std::vector<cv::Rect> boxes_;
    std::vector<float> scores_;
    std::vector<int> classIds_;
    
    int inputHeight_;
    int inputWidth_;
    std::vector<uint16_t> h_input_;
    std::vector<uint16_t> h_output_;
    cv::Mat resized_;
    std::vector<cv::Mat> channels_;

    // GPU-resident buffers for CUDA pipeline
    cv::cuda::GpuMat gpu_resized_;
    cv::cuda::GpuMat gpu_normalized_;
    std::vector<cv::cuda::GpuMat> gpu_channels_;
};