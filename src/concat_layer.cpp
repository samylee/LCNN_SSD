#include "concat_layer.h"

ConcatLayer::ConcatLayer()
{
}

ConcatLayer::~ConcatLayer()
{
}

void ConcatLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const int num_axes = 4;
	concat_axis_ = concat_param_.concat_axis_;

	// Initialize with the first blob.
	vector<int> top_shape;
	top_shape.push_back(bottom_data_[0].shape(0));
	top_shape.push_back(bottom_data_[0].shape(1));
	top_shape.push_back(bottom_data_[0].shape(2));
	top_shape.push_back(bottom_data_[0].shape(3));

	num_concats_ = bottom_data_[0].count(0, concat_axis_);
	concat_input_size_ = bottom_data_[0].count(concat_axis_ + 1, concat_axis_ + 1);
	int bottom_count_sum = bottom_data_[0].counts_;
	for (int i = 1; i < bottom_data_.size(); ++i) {
		for (int j = 0; j < num_axes; ++j) {
			if (j == concat_axis_) { continue; }
		}
		bottom_count_sum += bottom_data_[0].counts_;
		top_shape[concat_axis_] += bottom_data_[i].shape(concat_axis_);
	}
	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
}

void ConcatLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	float* top_data = top_data_[0].cpu_data();
	int offset_concat_axis = 0;
	const int top_concat_axis = top_data_[0].shape(concat_axis_);
	for (int i = 0; i < bottom_data_.size(); ++i) {
		const float* bottom_data = bottom_data_[i].cpu_data();
		const int bottom_concat_axis = bottom_data_[i].shape(concat_axis_);
		for (int n = 0; n < num_concats_; ++n) {
			caffe_copy(bottom_concat_axis * concat_input_size_,
				bottom_data + n * bottom_concat_axis * concat_input_size_,
				top_data + (n * top_concat_axis + offset_concat_axis)
				* concat_input_size_);
		}
		offset_concat_axis += bottom_concat_axis;
	}
}