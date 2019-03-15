#include "normalize_layer.h"

NormalizeLayer::NormalizeLayer()
{
}

NormalizeLayer::~NormalizeLayer()
{
	this->weights_.release();
	this->norm_.release();
	this->sum_channel_multiplier_.release();
	this->sum_spatial_multiplier_.release();
	this->buffer_.release();
	this->buffer_channel_.release();
	this->buffer_spatial_.release();
}

void NormalizeLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_) {
	buffer_.reshape(1, bottom_data_[0].channel_, bottom_data_[0].height_, bottom_data_[0].width_);
	buffer_channel_.reshape(1, bottom_data_[0].channel_, 1, 1);
	buffer_spatial_.reshape(1, 1, bottom_data_[0].height_, bottom_data_[0].width_);
	norm_.reshape(bottom_data_[0].num_, 1, bottom_data_[0].height_, bottom_data_[0].width_);
	eps_ = 1e-010f;;
	int channels = bottom_data_[0].channel_;
	int spatial_dim = bottom_data_[0].width_ * bottom_data_[0].height_;
	sum_channel_multiplier_.reshape(1, channels, 1, 1);
	caffe_set(channels, 1.0, sum_channel_multiplier_.cpu_data());
	sum_spatial_multiplier_.reshape(
		1, 1, bottom_data_[0].height_, bottom_data_[0].width_);
	caffe_set(spatial_dim, 1.0, sum_spatial_multiplier_.cpu_data());

	top_data_.resize(1);
	top_data_[0].reshape(bottom_data_[0].num_, bottom_data_[0].channel_, bottom_data_[0].height_, bottom_data_[0].width_);
	buffer_.reshape(1, bottom_data_[0].channel_,
		bottom_data_[0].height_, bottom_data_[0].width_);

	norm_.reshape(bottom_data_[0].num_, 1, bottom_data_[0].height_, bottom_data_[0].width_);

	if (spatial_dim != sum_spatial_multiplier_.counts_) {
		sum_spatial_multiplier_.reshape(
			1, 1, bottom_data_[0].height_, bottom_data_[0].width_);
		caffe_set(spatial_dim, float(1),
			sum_spatial_multiplier_.cpu_data());
		buffer_spatial_.reshape(1, 1, bottom_data_[0].height_, bottom_data_[0].width_);
	}
}

void NormalizeLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_) {
	const float* bottom_data = bottom_data_[0].cpu_data();
	float* top_data = top_data_[0].cpu_data();
	const float* scale = this->weights_.cpu_data();
	float* buffer_data = buffer_.cpu_data();
	float* norm_data = norm_.cpu_data();
	// add eps to avoid overflow
	caffe_set(norm_.counts_, float(eps_), norm_data);
	const float* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
	const float* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
	int num = bottom_data_[0].num_;
	int dim = bottom_data_[0].counts_ / num;
	int spatial_dim = bottom_data_[0].height_ * bottom_data_[0].width_;
	int channels = bottom_data_[0].channel_;
	for (int n = 0; n < num; ++n) {
		caffe_sqr(dim, bottom_data, buffer_data);
		caffe_cpu_gemv(CblasTrans, channels, spatial_dim, float(1),
			buffer_data, sum_channel_multiplier, float(1),
			norm_data);
		// compute norm
		caffe_powx(spatial_dim, norm_data, float(0.5), norm_data);
		// scale the layer
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
			1, float(1), sum_channel_multiplier, norm_data,
			float(0), buffer_data);
		caffe_div(dim, bottom_data, buffer_data, top_data);
		norm_data += spatial_dim;

		// scale the output
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
			1, float(1), scale, sum_spatial_multiplier,
			float(0),
			buffer_data);
		caffe_mul(dim, top_data, buffer_data, top_data);

		bottom_data += dim;
		top_data += dim;
	}
}