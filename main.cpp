#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

vector<Point> points;
int nbclic = 0;



void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		nbclic++;
		points.push_back(Point(x, y));

		if (nbclic == 4) {

			cv::setMouseCallback("Result", NULL, NULL);

		}

	}
}





int main()
{

	
	

	//setMouseCallback("Image Depart", on_mouse, NULL);



    string filename = "/Users/leo-d/Desktop/videotest";
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
        return match1.distance < match2.distance;
    };

    cv::namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;

        orbfd->detect(frame, keypoints);
        orbfd->compute(frame, keypoints, descriptors);
        if (!lastDescriptors.empty()) {
            bfm->match(descriptors, lastDescriptors, matches);
        }
        std::sort(matches.begin(), matches.end(), matchCompare);

        if(frame.empty())
            break;
        cv::Mat kpframe;
        if (!lastFrame.empty() && !matches.empty() && !lastDescriptors.empty())
            cv::drawMatches(frame, keypoints, lastFrame, lastKeypoints, matches, kpframe);
        else
            cv::drawKeypoints(frame, keypoints, kpframe);
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







