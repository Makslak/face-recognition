#pragma once

#include "../BaseFaceRecognizer/BaseFaceRecognizer.hpp"

class DNNRecognizer : public BaseFaceRecognizer
{
public:

    DNNRecognizer(const std::string& dnnPath);
    void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) override;
    void read(const std::string& path) override;
    void write(const std::string& path);
    std::pair<int, float> predict(const cv::Mat& frame) override;

private:

    cv::Mat _getEmbedding(const cv::Mat& face);

    cv::dnn::Net _net;
    std::vector<cv::Mat> _embeddings;
};