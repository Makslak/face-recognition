#pragma once

#include "../BaseFaceRecognizer/BaseFaceRecognizer.hpp"


/* Face recognizer based on cv::dnn::Net. */
class DNNRecognizer : public BaseFaceRecognizer
{
public:

    /* Create object with model weights. */
    DNNRecognizer(const std::string& dnnPath);

    /* Train a model on the given face images and labels */
    void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) override;

    /* Reads pretrained embeddings */
    void read(const std::string& path) override;

    /* Saves current embeddings */
    void write(const std::string& path) const;

    /**
     * Predict face from image of face.
     *
     * @param frame  Input face image.
     * @return Pair {labelId, confidence}.
     */
    std::pair<int, float> predict(const cv::Mat& frame) override;

private:

    cv::Mat _getEmbedding(const cv::Mat& face);

    cv::dnn::Net _net;
    std::vector<cv::Mat> _embeddings;
};