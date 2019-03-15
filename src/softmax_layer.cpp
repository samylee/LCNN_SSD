#include "softmax_layer.h"

SoftmaxLayer::SoftmaxLayer()
{
}

SoftmaxLayer::~SoftmaxLayer()
{
	this->sum_multiplier_.release();
	this->scale_.release();
}

void SoftmaxLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	softmax_axis_ = 2;
	top_data_.resize(1);
	top_data_[0].reshape(bottom_data_[0].num_, bottom_data_[0].channel_, bottom_data_[0].height_, bottom_data_[0].width_);

	vector<int> mult_dims(1, bottom_data_[0].shape(softmax_axis_));
	sum_multiplier_.reshape(mult_dims[0], 1, 1, 1);
	float* multiplier_data = sum_multiplier_.cpu_data();
	caffe_set(sum_multiplier_.counts_, float(1), multiplier_data);
	outer_num_ = bottom_data_[0].count(0, softmax_axis_);
	inner_num_ = bottom_data_[0].count(softmax_axis_ + 1, softmax_axis_ + 1);
	vector<int> scale_dims{ bottom_data_[0].num_, bottom_data_[0].channel_, bottom_data_[0].height_,bottom_data_[0].width_ };
	scale_dims[softmax_axis_] = 1;
	scale_.reshape(scale_dims[0], scale_dims[1], scale_dims[2], scale_dims[3]);
}
void SoftmaxLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const float* bottom_data = bottom_data_[0].cpu_data();
	float* top_data = top_data_[0].cpu_data();
	float* scale_data = scale_.cpu_data();
	int channels = bottom_data_[0].shape(softmax_axis_);
	int dim = bottom_data_[0].counts_ / outer_num_;
	caffe_copy(bottom_data_[0].counts_, bottom_data, top_data);
	// We need to subtract the max to avoid numerical issues, compute the exp,
	// and then normalize.
	for (int i = 0; i < outer_num_; ++i) {
		// initialize scale_data to the first plane
		caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
		for (int j = 0; j < channels; j++) {
			for (int k = 0; k < inner_num_; k++) {
				scale_data[k] = std::max(scale_data[k],
					bottom_data[i * dim + j * inner_num_ + k]);
			}
		}
		// subtraction
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num_,
			1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
		// exponentiation
		caffe_exp(dim, top_data, top_data);
		// sum after exp
		caffe_cpu_gemv(CblasTrans, channels, inner_num_, 1.,
			top_data, sum_multiplier_.cpu_data(), 0., scale_data);
		// division
		for (int j = 0; j < channels; j++) {
			caffe_div(inner_num_, top_data, scale_data, top_data);
			top_data += inner_num_;
		}
	}
}