#ifndef BLOB_H
#define BOLB_H

#include <vector>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "math_functions.h"

using namespace std;
using namespace cv;

class Blob
{
public:
	Blob();

	void reshape(int num, int channel, int height, int width);
	void set_data(float *data);
	void release();

	//返回data值
	float* cpu_data();

	//返回nchw中某个尺度
	int shape(int index);

	int count(int start_axis, int end_axis);
	int offset(int n, int c = 0, int h = 0, int w = 0);

	int num_;
	int channel_;
	int height_;
	int width_;
	int counts_;

private:
	float *data_;
};

#endif