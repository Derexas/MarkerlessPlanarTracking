#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;

int main()
{
    string filename = "/home/black/Video/Futurama.S05E14.mkv";
    cv::VideoCapture capture(filename);
    cv::Mat frame, lastFrame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Ptr<cv::ORB> orbfd = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints, lastKeypoints;
    cv::Mat descriptors, lastDescriptors;
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create();

    std::vector<cv::DMatch> matches;

    auto matchCompare = [](const cv::DMatch& match1, const cv::DMatch& match2) {
        return cv::norm(keypoints[match1.queryIdx].pt.;
    };

    cv::namedWindow( "w", 1);
    for( ; ; )
    {
        cv::Mat kpframe;
        capture >> frame;

        orbfd->detect(frame, keypoints);
        orbfd->compute(frame, keypoints, descriptors);
        if (!lastDescriptors.empty()) {
            bfm->match(descriptors, lastDescriptors, matches);
            if (!matches.empty()) {
                std::vector<cv::Point2f> srcPoints, dstPoints;
                for (auto m = matches.begin(); m != matches.end(); m++) {
                    srcPoints.push_back(keypoints[m->queryIdx].pt);
                    dstPoints.push_back(keypoints[m->trainIdx].pt);
                }
                cv::Mat homography = cv::findHomography(srcPoints, dstPoints, cv::RANSAC);

                std::sort(matches.begin(), matches.end(), matchCompare);

                if(frame.empty())
                    break;
                if (!lastFrame.empty() && !matches.empty() && !lastDescriptors.empty()) {
                    int size = matches.size();
                    int n = std::min(20, size);
                    std::vector<cv::DMatch>::const_iterator first = matches.begin();
                    std::vector<cv::DMatch>::const_iterator last = matches.begin() + n;
                    std::vector<cv::DMatch> nearestMatches(first, last);
                    cv::drawMatches(frame, keypoints, lastFrame, lastKeypoints, nearestMatches, kpframe);
                }
            }
        }
        if (kpframe.empty())
            kpframe = frame;//cv::drawKeypoints(frame, keypoints, kpframe);
        lastFrame = frame;
        lastKeypoints = keypoints;
        lastDescriptors = descriptors;
        cv::imshow("w", kpframe);
        cv::waitKey(2); // waits to display frame
    }
    cv::waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
    return 0;
}
