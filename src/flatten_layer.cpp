#include "flatten_layer.h"

FlattenLayer::FlattenLayer()
{
}

FlattenLayer::~FlattenLayer()
{
}

void FlattenLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const int start_axis = 1;
	const int end_axis = 3;
	vector<int> top_shape;
	for (int i = 0; i < start_axis; ++i) {
		top_shape.push_back(bottom_data_[0].shape(i));
	}
	const int flattened_dim = bottom_data_[0].count(start_axis, end_axis + 1);
	top_shape.push_back(flattened_dim);
	for (int i = end_axis + 1; i < 4; ++i) {
		top_shape.push_back(bottom_data_[0].shape(i));
	}
	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], 1, 1);
}
void FlattenLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	float* bottom_data = bottom_data_[0].cpu_data();
	float* top_data = top_data_[0].cpu_data();

	const int top_count = top_data_[0].counts_;
	for (int i = 0; i < top_count; i++) {
		top_data[i] = bottom_data[i];
	}
}