#include "reshape_layer.h"

ReshapeLayer::ReshapeLayer()
{
}

ReshapeLayer::~ReshapeLayer()
{
}

void ReshapeLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	int output_num = 21;
	int channels_ = bottom_data_[0].counts_ / output_num;
	vector<int> top_shape;
	top_shape.push_back(1);
	top_shape.push_back(channels_);
	top_shape.push_back(output_num);
	top_shape.push_back(1);

	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
}

void ReshapeLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	float* bottom_data = bottom_data_[0].cpu_data();
	float* top_data = top_data_[0].cpu_data();

	const int top_count = top_data_[0].counts_;
	for (int i = 0; i < top_count; i++) {
		top_data[i] = bottom_data[i];
	}
}