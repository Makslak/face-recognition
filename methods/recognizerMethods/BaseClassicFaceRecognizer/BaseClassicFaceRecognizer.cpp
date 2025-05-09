#include "BaseClassicFaceRecognizer.hpp"


void BaseClassicFaceRecognizer::train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string &savePath)
{
    std::vector<cv::Mat> images;
    cv::Mat temp;

    for (size_t i = 0; i < faces.size(); ++i)
    {
        cv::cvtColor(faces[i], temp, cv::COLOR_BGR2GRAY);
        cv::resize(temp, temp, {100, 100});
        images.push_back(temp.clone());
    }
    
    this->_model->train(images, facesLabels);
    this->_model->save(savePath);
}


void BaseClassicFaceRecognizer::read(const std::string &path){
    this->_model->read(path);
}


std::pair<int, float> BaseClassicFaceRecognizer::predict(const cv::Mat &frame)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, {100, 100});

    int predictedLabel = -1;
    double confidence = 0.0;
    this->_model->predict(gray, predictedLabel, confidence);

    return std::make_pair(predictedLabel, static_cast<float>(confidence));
}
