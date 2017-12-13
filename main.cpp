#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

int main()
{
    cv::Mat image;
    cout << "Hello World!" << endl;
    string filename = "/home/black/Video/Futurama.S05E14.mkv";
    cv::VideoCapture capture(filename);
    cout << "Hello World!" << endl;
    cv::Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Ptr<cv::FastFeatureDetector> fastfd = cv::FastFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;

    cv::namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;

        cout << "Hello World!" << endl;
        fastfd->detect(frame, keypoints);
        cout << "Hello World!" << endl;
        cv::Mat kpframe;
        cout << "Hello World!" << endl;
        cv::drawKeypoints(frame, keypoints, kpframe);
        cout << "Hello World!" << endl;

        if(kpframe.empty())
            break;
        cv::imshow("w", kpframe);
        cv::waitKey(20); // waits to display frame
    }
    cv::waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
    return 0;
}
