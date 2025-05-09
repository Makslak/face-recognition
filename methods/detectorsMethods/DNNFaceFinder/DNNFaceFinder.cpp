#include "DNNFaceFinder.hpp"


void DNNFaceFinder::read(const std::string &path){
    this->_Net = cv::dnn::readNetFromONNX(path);
}

void DNNFaceFinder::read(const std::string &path, bool opencl)
{
    this->read(path);
    if(!opencl) return;

    this->_Net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    this->_Net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

void DNNFaceFinder::find(const cv::Mat &frame, float confThreshold)
{
    cv::Size size = {640, 640};
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, size, {0, 0, 0}, true, false);
    this->_Net.setInput(blob);
    cv::Mat output = this->_Net.forward();

    this->faces.clear();
    this->confidences.clear();

    int outputSize = 8400;

    float confidence;
    int x, y, width, height;
    float cx, cy, w, h;

    float* data = output.ptr<float>();

    for (size_t i = 0; i < outputSize; ++i)
    {
        confidence = data[4 * outputSize + i];
        if(confidence < confThreshold) continue;

        cx = data[i];
        cy = data[outputSize + i];
        w = data[2 * outputSize + i];
        h = data[3 * outputSize + i];

        width = static_cast<int>((w * frame.cols) / size.width);
        height = static_cast<int>((h * frame.rows) / size.height);
        x = static_cast<int>((cx * frame.cols) / size.width) - width / 2;
        y = static_cast<int>((cy * frame.rows) / size.height) - height / 2;

        if((x + width) > frame.cols) {width = frame.cols - x - 1;}
        if((y + height) > frame.rows) {height = frame.rows - y - 1;}

        this->faces.push_back(cv::Rect(x, y, width, height));
        this->confidences.push_back(confidence);
    }

    float nmsThreshold = 0.4;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(this->faces, this->confidences, confThreshold, nmsThreshold, indices);

    std::vector<cv::Rect> finalFaces;
    std::vector<float> finalConfidences;
    for (int idx : indices)
    {
        finalFaces.push_back(this->faces[idx]);
        finalConfidences.push_back(this->confidences[idx]);
    }

    this->faces = finalFaces;
    this->confidences = finalConfidences;
}
