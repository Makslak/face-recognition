#include "BaseLandmarksFinder.hpp"

cv::Rect BaseLandmarksFinder::getExtendedRect(const cv::Mat& frame, const cv::Rect &rect)
{
    cv::Rect res = rect;

    (res.x > 50) ? res.x -= 50 : res.x = 0;
    (res.y > 50) ? res.y -= 50 : res.y = 0;
    (res.x + res.width + 100 > frame.cols) ? res.width = (frame.cols - (res.x + 1)) : res.width += 100;
    (res.y + res.height + 100 > frame.rows) ? res.height = (frame.rows - (res.y + 1)) : res.height += 100;

    // std::cout << frame.size << "\n";

    // std::cout << "[" << res.x << "," <<  res.y << "," << res.x + res.width << "," << res.y + res.height << "]" << "\n";

    return res;
}