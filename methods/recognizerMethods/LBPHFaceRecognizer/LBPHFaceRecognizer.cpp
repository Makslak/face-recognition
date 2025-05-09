#include "LBPHFaceRecognizer.hpp"

LBPHFaceRecognizer::LBPHFaceRecognizer(){
    this->_model = cv::face::LBPHFaceRecognizer::create();
}