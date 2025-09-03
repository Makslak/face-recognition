#include "DNNLandmarksFinder.hpp"

void DNNLandmarksFinder::read(const std::string &path){
    this->_net = cv::dnn::readNetFromONNX(path);
}

void DNNLandmarksFinder::find(const cv::Mat &frame, const cv::Rect &face)
{
    this->landmarks.clear();

    const cv::Rect roi = this->getExtendedRect(frame, face);
    cv::Mat resized = frame(roi);
    cv::resize(resized, resized, {256, 256});

    const cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, {256, 256}, {0, 0, 0}, true, false);
    this->_net.setInput(blob);

    cv::Mat output = this->_net.forward();

    const int numLandmarks = output.size[1];
    const int heatmapHeight = output.size[2];
    const int heatmapWidth = output.size[3];

    for (int i = 0; i < numLandmarks; i++) 
    {
        cv::Mat heatmap(heatmapHeight, heatmapWidth, CV_32F, output.ptr(0, i));
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmap, &minVal, &maxVal, &minLoc, &maxLoc);
        
        float x = static_cast<float>(maxLoc.x) / static_cast<float>(heatmapWidth);
        float y = static_cast<float>(maxLoc.y) / static_cast<float>(heatmapHeight);
        
        landmarks.emplace_back(static_cast<int>(roi.x + x * roi.width), static_cast<int>(roi.y + y * roi.height));
    }
}
