#pragma once

#include <opencv2/opencv.hpp>

class BaseFaceRecognizer
{
public:

    virtual void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) = 0;
    virtual void read(const std::string& path) = 0;
    virtual std::pair<int, float> predict(const cv::Mat& frame) = 0;
};