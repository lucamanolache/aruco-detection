#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/aruco.hpp"

const auto DICT = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);

int main() {
    return 0;
}

void getTranslation(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::aruco::Dictionary &dictionary, cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    cv::Mat rvec_mat, tvec_mat;
    cv::aruco::detectMarkers(frame, DICT, corners, ids);
    if (ids.size() > 0) {
        cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvec_mat, tvec_mat);
        rvec = rvec_mat;
        tvec = tvec_mat;
    }
}
