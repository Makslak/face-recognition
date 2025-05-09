#pragma once

#include <opencv2/opencv.hpp>

class FaceReconstraction
{
public:
    
    void read(std::string landmarksPath, std::string depthPath);
    void getPoints(const cv::Mat& frame, cv::Rect face);

    std::vector<cv::Point3f> points;

private:

    cv::Rect _getExtendedRect(const cv::Mat& frame, const cv::Rect &rect);

    cv::dnn::Net _landmarksNet;
    cv::dnn::Net _depthNet;
};