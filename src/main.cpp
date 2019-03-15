#include "lcnn.h"
#include <time.h>

void loadRandomColor(std::vector<std::vector<int> > &classes_color)
{
	//随机生成各种类颜色
	srand((unsigned)time(NULL));
	int small = 100;
	int big = 255;

	for (int i = 0; i < 20; i++)
	{
		int red = (rand() % (big - small + 1)) + small;
		int green = (rand() % (big - small + 1)) + small;
		int blue = (rand() % (big - small + 1)) + small;

		std::vector<int> cls_color;
		cls_color.push_back(red);
		cls_color.push_back(green);
		cls_color.push_back(blue);

		classes_color.push_back(cls_color);
	}
}

void drawImage(Mat &image, vector<vector<float> > &detections, float detect_thresh, char *classes[], std::vector<std::vector<int> > &classes_color)
{
	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		const float score = d[2];
		if (score >= detect_thresh) {
			int left = d[3] * image.cols;
			int top = d[4] * image.rows;
			int right = d[5] * image.cols;
			int bottom = d[6] * image.rows;
			int clslabel = d[1] - 1;

			std::string clsclass = classes[clslabel];
			int r = classes_color[clslabel][0];
			int g = classes_color[clslabel][1];
			int b = classes_color[clslabel][2];

			cv::rectangle(image, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(r, g, b), 2);
			cv::putText(image, clsclass, cv::Point(left, top + 15), 0, 0.5, cv::Scalar(r, g, b), 2);
		}
	}
}

int main()
{
	//VOC种类
	char *classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle",
		"bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person", 
		"pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	std::vector<std::vector<int> > classes_color;
	loadRandomColor(classes_color);

	//init model
	string model_path = "model/ssd.dat";
	float detecte_thresh = 0.3;
	Lcnn lcnn(model_path);

	string file_type = "image";
	if (file_type == "image")
	{
		Mat image = imread("000004.jpg");
		if (image.empty()) 
			return -1;

		std::vector<vector<float> > detections = lcnn.detect(image);
		drawImage(image, detections, detecte_thresh, classes, classes_color);

		imshow("out", image);
		waitKey(0);
	}
	else
	{
		VideoCapture cap(0);
		if (!cap.isOpened()) return -1;
		Mat image;

		while (true)
		{
			cap >> image;
			if (image.empty()) break;

			clock_t start_t = clock();
			std::vector<vector<float> > detections = lcnn.detect(image);
			drawImage(image, detections, detecte_thresh, classes, classes_color);
			cout << "Cost time: " << clock() - start_t << endl;

			imshow("out", image);
			if (waitKey(1) > 0)
				break;
		}
		cap.release();
	}

	return 0;
}