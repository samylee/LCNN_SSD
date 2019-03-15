#include "blob.h"

Blob::Blob()
{
	this->num_ = 0;
	this->channel_ = 0;
	this->height_ = 0;
	this->width_ = 0;

	this->counts_ = 0;
	data_ = NULL;
}

void Blob::reshape(int num, int channel, int height, int width)
{
	this->num_ = num;
	this->channel_ = channel;
	this->height_ = height;
	this->width_ = width;

	this->counts_ = this->num_ * this->channel_ * this->height_ * this->width_;

	if (data_ != NULL)
	{
		free(data_);
	}

	data_ = (float*)malloc(sizeof(float) * counts_);
}

//将数据存放进blob
void Blob::set_data(float* data)
{
	caffe_copy(this->counts_, data, this->data_);
}

void Blob::release()
{
	if (data_ != NULL)
	{
		free(data_);
		data_ = NULL;
	}
}

float* Blob::cpu_data()
{
	return data_;
}

int Blob::shape(int index)
{
	switch (index)
	{
	case 0:
		return this->num_;
		break;
	case 1:
		return this->channel_;
		break;
	case 2:
		return this->height_;
		break;
	case 3:
		return this->width_;
		break;
	}
}

int Blob::count(int start_axis, int end_axis)
{
	int count = 1;
	for (int i = start_axis; i < end_axis; ++i) {
		count *= shape(i);
	}
	return count;
}

int Blob::offset(int n, int c, int h, int w)
{
	return ((n * this->channel_ + c) * this->height_ + h) * this->width_ + w;
}