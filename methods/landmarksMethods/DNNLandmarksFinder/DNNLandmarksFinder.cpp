#include "DNNLandmarksFinder.hpp"

void DNNLandmarksFinder::read(const std::string &path){
    this->_net = cv::dnn::readNetFromONNX(path);
}

void DNNLandmarksFinder::find(const cv::Mat &frame, const cv::Rect &face)
{
    this->landmarks.clear();

    cv::Rect roi = this->getExtendedRect(frame, face);
    cv::Mat resised = frame(roi);
    cv::resize(resised, resised, {256, 256});

    cv::Mat blob = cv::dnn::blobFromImage(resised, 1.0 / 255.0, {256, 256}, {0, 0, 0}, true, false);
    this->_net.setInput(blob);

    cv::Mat output = this->_net.forward();

    int numLandmarks = output.size[1];
    int heatmapHeight = output.size[2];
    int heatmapWidth = output.size[3];

    for (int i = 0; i < numLandmarks; i++) 
    {
        cv::Mat heatmap(heatmapHeight, heatmapWidth, CV_32F, output.ptr(0, i));
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmap, &minVal, &maxVal, &minLoc, &maxLoc);
        
        float x = static_cast<float>(maxLoc.x) / heatmapWidth;
        float y = static_cast<float>(maxLoc.y) / heatmapHeight;
        
        x = static_cast<int>(roi.x + x * roi.width);
        y = static_cast<int>(roi.y + y * roi.height);
        
        landmarks.push_back(cv::Point2f(x, y));
    }
}
