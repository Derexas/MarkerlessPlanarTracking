#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;

vector<cv::Point2d> imagePoints;
int nbclic = 0;
cv::Size size;
cv::Mat image;
string filename = "/home/black/School/SFX/MarkerlessPlanarTracking/hxh.mp4";

void homographyVideo();
void homographyImages();
void computePose(vector<cv::Point2d> pts, cv::Mat im);

void on_mouse(int e, int x, int y, int d, void *ptr)
{
    if (e == cv::EVENT_LBUTTONDOWN)
    {
        cout << "click" << endl;
        nbclic++;
        imagePoints.push_back(cv::Point(x, y));
        cv::Mat frame = image;
        for (cv::Point2d p : imagePoints) {
            cv::circle(frame, p, 5, cv::Scalar( 0, 0, 255 ));
        }
        cv::imshow("w", frame);
        if (nbclic == 4) {
            cv::setMouseCallback("Result", NULL, NULL);
            //computePose(imagePoints, image);
            //homographyVideo();
            homographyImages();
        }
    }
}

int main()
{
    cv::VideoCapture capture(filename);
    cv::Mat frame, lastFrame;

    capture >> frame;
    size = cv::Size(frame.cols, frame.rows);
    image = frame;

    cv::imshow("w", frame);
    cv::setMouseCallback("w", on_mouse, NULL);
    while(nbclic != 4) {
        cv::waitKey(20); // waits to display frame
    }
    cout << "done" << endl;
}

void homographyImages() {
    cout << "1" << endl;
    cv::VideoCapture capture(filename);
    cv::Mat frame1, frame2, kpframe;

    cv::Ptr<cv::ORB> orbfd = cv::ORB::create();
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;

    capture >> frame1;
    for (int i = 0; i < 20; i++) capture >> frame2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orbfd->detect(frame1, keypoints1);
    orbfd->compute(frame1, keypoints1, descriptors1);
    while (true) {
        capture >> frame2;
        orbfd->detect(frame2, keypoints2);
        orbfd->compute(frame2, keypoints2, descriptors2);

        bfm->match(descriptors1, descriptors2, matches);
        std::vector<cv::Point2d> srcPoints, dstPoints;
        for (auto m = matches.begin(); m != matches.end(); m++) {
            srcPoints.push_back(keypoints1[m->queryIdx].pt);
            dstPoints.push_back(keypoints2[m->trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat homography = cv::findHomography(srcPoints, dstPoints, mask, cv::RANSAC, 5.0);

        cv::drawMatches(frame1, keypoints1, frame2, keypoints2, matches, kpframe);
        kpframe = frame2;
        //cv::warpPerspective(frame1, kpframe, homography, cv::Size(frame2.cols, frame2.rows));
        vector<cv::Point3d> pp;
        for (cv::Point2d p : imagePoints)
            pp.push_back(cv::Point3d(p.x, p.y, 1));
        cv::transform(pp, pp, homography);
        for (cv::Point3d p : pp) {
            cv::circle(kpframe, cv::Point(p.x/p.z, p.y/p.z), 5, cv::Scalar( 0, 0, 255 ));
        }
        cv::imshow("test", kpframe);
        cv::waitKey(20);
    }
}

void homographyVideoOld() {
    cv::VideoCapture capture(filename);
    cv::Mat frame, lastFrame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Ptr<cv::ORB> orbfd = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints, lastKeypoints;
    cv::Mat descriptors, lastDescriptors;
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;

    //std::vector<double> distances;

    auto matchCompare = [](const cv::DMatch& match1, const cv::DMatch& match2) {
        return match1.distance < match2.distance;
    };

    cv::Mat referenceFrame;
    capture >> referenceFrame;
    std::vector<cv::KeyPoint> referenceKeypoints;
    cv::Mat referenceDescriptors;
    orbfd->detect(referenceFrame, referenceKeypoints);
    orbfd->compute(referenceFrame, referenceKeypoints, referenceDescriptors);

    cv::namedWindow( "w", 1);
    for( ; ; )
    {
        cv::Mat kpframe;
        capture >> frame;

        orbfd->detect(frame, keypoints);
        orbfd->compute(frame, keypoints, descriptors);
        if (!referenceDescriptors.empty()) {
            bfm->match(referenceDescriptors, descriptors, matches);
            if (!matches.empty()) {
                int size = matches.size();
                int n = size;//std::min(40, size);
                std::sort(matches.begin(), matches.end(), matchCompare);

                std::vector<cv::Point2d> srcPoints, dstPoints;
                for (auto m = matches.begin(); m != matches.begin()+n; m++) {
                    srcPoints.push_back(referenceKeypoints[m->queryIdx].pt);
                    dstPoints.push_back(keypoints[m->trainIdx].pt);
                }

                cv::Mat mask;
                cv::Mat homography = cv::findHomography(srcPoints, dstPoints, mask, cv::RANSAC, 5.0);
                cout << homography << endl;
                // convert mask to vector for display
                vector<char> matchMask(n, 1);
                matchMask.assign(mask.datastart, mask.dataend);

                /*distances.erase(distances.begin(), distances.end());
                for (auto m = matches.begin(); m != matches.end(); m++) {
                    m->distance = cv::norm(keypoints[m->queryIdx].pt - lastKeypoints[m->trainIdx].pt);
                }*/


                if(frame.empty())
                    break;
                if (!lastFrame.empty() && !matches.empty() && !lastDescriptors.empty()) {
                    std::vector<cv::DMatch>::const_iterator first = matches.begin();
                    std::vector<cv::DMatch>::const_iterator last = matches.begin() + n;
                    std::vector<cv::DMatch> nearestMatches(first, last);
                    cv::Scalar colors = cv::Scalar::all(-1);
                    cv::drawMatches(frame, keypoints, referenceFrame, referenceKeypoints, nearestMatches, kpframe, colors, colors, matchMask);
                    //kpframe = frame;
                    cv::warpPerspective(referenceFrame, kpframe, homography, cv::Size(frame.cols, frame.rows));
                    vector<cv::Point3d> pp;
                    for (cv::Point2d p : imagePoints)
                        pp.push_back(cv::Point3d(p.x, p.y, 1));
                    cv::transform(pp, pp, homography);
                    for (cv::Point3d p : pp) {
                        cv::circle(kpframe, cv::Point(p.x, p.y), 5, cv::Scalar( 0, 0, 255 ));
                    }
                }
            }
        }
        if (kpframe.empty())
            kpframe = frame;//cv::drawKeypoints(frame, keypoints, kpframe);
        cv::imshow("w", kpframe);
        cv::waitKey(20); // waits to display frame
        lastFrame = frame;
        lastKeypoints = keypoints;
        lastDescriptors = descriptors;
    }
    cv::waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
}
