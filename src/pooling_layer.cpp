#include "pooling_layer.h"

PoolingLayer::PoolingLayer()
{

}

PoolingLayer::~PoolingLayer()
{

}

void PoolingLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	//pooling 层输入输出都是一个
	bottom_data_.resize(1);
	top_data_.resize(1);

	this->global_pooling_ = this->pool_param_.global_pooling_;
	if (global_pooling_) {
		kernel_h_ = bottom_data_[0].height_;
		kernel_w_ = bottom_data_[0].width_;
	}
	else {
		if (this->pool_param_.has_kernel_size_) {
			kernel_h_ = kernel_w_ = this->pool_param_.kernel_size_;
		}
		else {
			kernel_h_ = this->pool_param_.kernel_h_;
			kernel_w_ = this->pool_param_.kernel_w_;
		}
	}

	if (!this->pool_param_.has_pad_h_) {
		pad_h_ = pad_w_ = this->pool_param_.pad_;
	}
	else {
		pad_h_ = this->pool_param_.pad_h_;
		pad_w_ = this->pool_param_.pad_w_;
	}
	if (!this->pool_param_.has_stride_h_) {
		stride_h_ = stride_w_ = this->pool_param_.stride_;
	}
	else {
		stride_h_ = this->pool_param_.stride_h_;
		stride_w_ = this->pool_param_.stride_w_;
	}

	channels_ = bottom_data_[0].channel_;
	height_ = bottom_data_[0].height_;
	width_ = bottom_data_[0].width_;

	pooled_height_ = static_cast<int>(ceil(static_cast<float>(
		height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
	pooled_width_ = static_cast<int>(ceil(static_cast<float>(
		width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
	if (pad_h_ || pad_w_) {
		// If we have padding, ensure that the last pooling starts strictly
		// inside the image (instead of at the padding); otherwise clip the last.
		if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
			--pooled_height_;
		}
		if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
			--pooled_width_;
		}
	}
	top_data_[0].reshape(bottom_data_[0].num_, channels_, pooled_height_, pooled_width_);
}

void PoolingLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	float* bottom_data = bottom_data_[0].cpu_data();	
	float* top_data = top_data_[0].cpu_data();
	int top_count = top_data_[0].counts_;
	switch (this->pool_param_.pool_method_) {
	case 0:
		caffe_set(top_count, float(-FLT_MAX), top_data);
		// The main loop
		for (int n = 0; n < bottom_data_[0].num_; ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int hstart = ph * stride_h_ - pad_h_;
						int wstart = pw * stride_w_ - pad_w_;
						int hend = min(hstart + kernel_h_, height_);
						int wend = min(wstart + kernel_w_, width_);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						const int pool_index = ph * pooled_width_ + pw;
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								const int index = h * width_ + w;
								if (bottom_data[index] > top_data[pool_index])
								{
									top_data[pool_index] = bottom_data[index];
								}
							}
						}
					}
				}
				// compute offset
				bottom_data += bottom_data_[0].offset(0, 1);
				top_data += top_data_[0].offset(0, 1);
			}
		}
		break;
	case 1:
		for (int i = 0; i < top_count; ++i) {
			top_data[i] = 0;
		}
		// The main loop
		for (int n = 0; n < bottom_data_[0].num_; ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int hstart = ph * stride_h_ - pad_h_;
						int wstart = pw * stride_w_ - pad_w_;
						int hend = min(hstart + kernel_h_, height_ + pad_h_);
						int wend = min(wstart + kernel_w_, width_ + pad_w_);
						int pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, height_);
						wend = min(wend, width_);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								top_data[ph * pooled_width_ + pw] +=
									bottom_data[h * width_ + w];
							}
						}
						top_data[ph * pooled_width_ + pw] /= pool_size;
					}
				}
				// compute offset
				bottom_data += bottom_data_[0].offset(0, 1);
				top_data += top_data_[0].offset(0, 1);
			}
		}
		break;
	default:
		break;
	}	
}