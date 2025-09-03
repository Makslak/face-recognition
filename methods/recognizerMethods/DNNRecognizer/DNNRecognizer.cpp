#include "DNNRecognizer.hpp"


DNNRecognizer::DNNRecognizer(const std::string &dnnPath){
    this->_net = cv::dnn::readNetFromONNX(dnnPath);
}


void DNNRecognizer::train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string &savePath)
{
    const int num = facesLabels.back();
    int iter = 0;
    this->_embeddings.clear();
    
    for (size_t i = 0; i <= num; ++i)
    {
        cv::Mat embedding = this->_getEmbedding(faces[iter]);
        ++iter;
        while (iter < facesLabels.size() && facesLabels[iter] == i)
        {
            embedding += this->_getEmbedding(faces[iter]);
            ++iter;
        }

        embedding /= cv::norm(embedding);
        this->_embeddings.push_back(embedding.clone());
    }
    this->write(savePath);
}


void DNNRecognizer::read(const std::string &path)
{
    this->_embeddings.clear();
    cv::FileStorage fs(path, cv::FileStorage::READ);
    const cv::FileNode node = fs["embeddings"];

    for (auto&& it : node)
    {
        cv::Mat emb;
        it >> emb;
        this->_embeddings.push_back(emb);
    }

    fs.release();
}

void DNNRecognizer::write(const std::string& path) const
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    
    fs << "embeddings" << "[";
    for (const auto & _embedding : this->_embeddings)
    {
        fs << _embedding;
    }
    fs << "]";
    
    fs.release();
}

std::pair<int, float> DNNRecognizer::predict(const cv::Mat &frame)
{
    cv::Mat embedding = this->_getEmbedding(frame);
    cv::transpose(embedding, embedding);
    std::pair<int, float> res = std::make_pair(-1, -1);

    for (size_t i = 0; i < this->_embeddings.size(); ++i)
    {
        float dot = cv::Mat(this->_embeddings[i] * embedding).at<float>(0, 0);
        if(dot > res.second){
            res = std::make_pair(i, dot);
        }
    }

    return res;
}


cv::Mat DNNRecognizer::_getEmbedding(const cv::Mat &face)
{
    cv::Mat resized;
    cv::resize(face, resized, {112, 112});
    const cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true, false);
    //cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 128.0, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);
    this->_net.setInput(blob);
    cv::Mat embedding = this->_net.forward();
    const double norm = cv::norm(embedding);
    embedding /= norm;
    return embedding;
}
