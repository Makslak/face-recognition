#include "KazemiLandmarksFinder.hpp"

KazemiLandmarksFinder::KazemiLandmarksFinder(){
    this->_facemark = cv::face::FacemarkKazemi::create();
}