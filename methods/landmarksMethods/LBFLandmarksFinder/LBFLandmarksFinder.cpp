#include "LBFLandmarksFinder.hpp"


LBFLandmarksFinder::LBFLandmarksFinder(){
    this->_facemark = cv::face::FacemarkLBF::create();
}