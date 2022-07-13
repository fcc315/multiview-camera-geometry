#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() // Groundtruth testing
{
	Mat imgL, imgR;
	imgL = imread("E:/phy-measure/CamCalib/data/tsukuba_l.png");
	imgR = imread("E:/phy-measure/CamCalib/data/tsukuba_r.png");

	cout << "Image Resolution: " << imgL.cols << " x " << imgL.rows << endl;

	int e1, e2;
	double dur;

	e1 = getTickCount();

	Mat descriptor_1, descriptor_2;
	vector<KeyPoint> keypoints_1, keypoints_2;

	Ptr<ORB> detector = ORB::create();
	vector<DMatch> matches;
	vector<DMatch> good_matches;
	BFMatcher matcher(NORM_L2);
	Ptr<DescriptorExtractor> extractor;

	detector->detectAndCompute(imgR, Mat(), keypoints_1, descriptor_1);
	detector->detectAndCompute(imgL, Mat(), keypoints_2, descriptor_2);
	matcher.match(descriptor_1, descriptor_2, matches);

	// Transform pixel coordinates between images
	//Mat homography = findHomography(kpt_1, kpt_2, CV_RANSAC);

	e2 = getTickCount();
	dur = (e2 - e1) / getTickFrequency();
	cout << "Dur (feature matching): " << dur << " s\n";


	Mat img_match;

	drawMatches(imgR, keypoints_1, imgL, keypoints_2, matches, img_match);
	imshow("Matches", img_match);
	waitKey();

	return 1;
}