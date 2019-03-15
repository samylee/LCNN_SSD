#include "ssd_common.h"
#include <algorithm>

void destroy_blob(vector<Blob>& des_datas_)
{
	for (int i = 0; i < des_datas_.size(); i++)
	{
		des_datas_[i].release();
	}
}

void destroy_blob_single(Blob& des_datas_)
{
	des_datas_.release();
}

void readToInputLayer(InputLayer &input, FILE* fp)
{
	fread(&input.input_params_.channel_, sizeof(int), 1, fp);
	fread(&input.input_params_.height_, sizeof(int), 1, fp);
	fread(&input.input_params_.width_, sizeof(int), 1, fp);
}

void readToConvLayer(ConvolutionLayer &conv, FILE* fp)
{
	int* pad_ = NULL;
	int* stride_ = NULL;
	int* kernel_size_ = NULL;
	int* dilation_ = NULL;
	fread(&conv.convolution_param_.pad_size_, sizeof(int), 1, fp);
	if (conv.convolution_param_.pad_size_ > 0)
	{
		pad_ = new int[conv.convolution_param_.pad_size_];
		fread(pad_, sizeof(int), conv.convolution_param_.pad_size_, fp);
		for (int i = 0; i < conv.convolution_param_.pad_size_; i++)
		{
			conv.convolution_param_.m_pad_.push_back(pad_[i]);
		}
	}

	fread(&conv.convolution_param_.stride_size_, sizeof(int), 1, fp);
	if (conv.convolution_param_.stride_size_ > 0)
	{
		stride_ = new int[conv.convolution_param_.stride_size_];
		fread(stride_, sizeof(int), conv.convolution_param_.stride_size_, fp);
		for (int i = 0; i < conv.convolution_param_.stride_size_; i++)
		{
			conv.convolution_param_.m_stride_.push_back(stride_[i]);
		}
	}

	fread(&conv.convolution_param_.kernel_size_size_, sizeof(int), 1, fp);
	if (conv.convolution_param_.kernel_size_size_ > 0)
	{
		kernel_size_ = new int[conv.convolution_param_.kernel_size_size_];
		fread(kernel_size_, sizeof(int), conv.convolution_param_.kernel_size_size_, fp);
		for (int i = 0; i < conv.convolution_param_.kernel_size_size_; i++)
		{
			conv.convolution_param_.m_kernel_size_.push_back(kernel_size_[i]);
		}
	}

	fread(&conv.convolution_param_.dilation_size_, sizeof(int), 1, fp);
	if (conv.convolution_param_.dilation_size_ > 0)
	{
		dilation_ = new int[conv.convolution_param_.dilation_size_];
		fread(dilation_, sizeof(int), conv.convolution_param_.dilation_size_, fp);
		for (int i = 0; i < conv.convolution_param_.dilation_size_; i++)
		{
			conv.convolution_param_.m_dilation_.push_back(dilation_[i]);
		}
	}

	fread(&conv.convolution_param_.has_pad_h_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.has_pad_w_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.pad_h_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.pad_w_, sizeof(int), 1, fp);

	fread(&conv.convolution_param_.has_kernel_h_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.has_kernel_w_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.kernel_h_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.kernel_w_, sizeof(int), 1, fp);

	fread(&conv.convolution_param_.has_stride_h_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.has_stride_w_, sizeof(bool), 1, fp);
	fread(&conv.convolution_param_.stride_h_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.stride_w_, sizeof(int), 1, fp);

	fread(&conv.convolution_param_.group_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.axis_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.num_output_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.force_nd_im2col_, sizeof(bool), 1, fp);

	fread(&conv.convolution_param_.weight_num_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.weight_channels_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.weight_height_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.weight_width_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.bias_num_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.bias_channels_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.bias_height_, sizeof(int), 1, fp);
	fread(&conv.convolution_param_.bias_width_, sizeof(int), 1, fp);

	int weight_count = conv.convolution_param_.weight_num_ * conv.convolution_param_.weight_channels_ * conv.convolution_param_.weight_height_ * conv.convolution_param_.weight_width_;
	int bias_count = conv.convolution_param_.bias_num_ * conv.convolution_param_.bias_channels_ * conv.convolution_param_.bias_height_ * conv.convolution_param_.bias_width_;
	float* weight = (float*)malloc(sizeof(float) * weight_count);
	float* bias = (float*)malloc(sizeof(float) * bias_count);
	fread(weight, sizeof(float), weight_count, fp);
	fread(bias, sizeof(float), bias_count, fp);

	//将权重和偏置矩阵赋值给blobs
	conv.weights_.reshape(conv.convolution_param_.weight_num_, conv.convolution_param_.weight_channels_, conv.convolution_param_.weight_height_, conv.convolution_param_.weight_width_);
	conv.weights_.set_data(weight);

	conv.bias_.reshape(conv.convolution_param_.bias_num_, conv.convolution_param_.bias_channels_, conv.convolution_param_.bias_height_, conv.convolution_param_.bias_width_);
	conv.bias_.set_data(bias);

	if (pad_ != NULL)
	{
		delete[] pad_;
	}
	if (stride_ != NULL)
	{
		delete[] stride_;
	}
	if (kernel_size_ != NULL)
	{
		delete[] kernel_size_;
	}
	if (dilation_ != NULL)
	{
		delete[] dilation_;
	}
	free(weight);
	free(bias);
}

void readToPoolingLayer(PoolingLayer &pool, FILE *fp)
{
	fread(&pool.pool_param_.global_pooling_, sizeof(bool), 1, fp);
	fread(&pool.pool_param_.pool_method_, sizeof(int), 1, fp);

	fread(&pool.pool_param_.has_kernel_size_, sizeof(bool), 1, fp);
	fread(&pool.pool_param_.kernel_size_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.kernel_h_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.kernel_w_, sizeof(int), 1, fp);

	fread(&pool.pool_param_.has_stride_h_, sizeof(bool), 1, fp);
	fread(&pool.pool_param_.stride_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.stride_h_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.stride_w_, sizeof(int), 1, fp);

	fread(&pool.pool_param_.has_pad_h_, sizeof(bool), 1, fp);
	fread(&pool.pool_param_.pad_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.pad_h_, sizeof(int), 1, fp);
	fread(&pool.pool_param_.pad_w_, sizeof(int), 1, fp);
}

void readToNormalizeLayer(NormalizeLayer &norm, FILE*fp)
{
	int num_ = 0;
	int channel_ = 0;
	int height_ = 0;
	int width_ = 0;

	fread(&num_, sizeof(int), 1, fp);
	fread(&channel_, sizeof(int), 1, fp);
	fread(&height_, sizeof(int), 1, fp);
	fread(&width_, sizeof(int), 1, fp);

	int weight_count = num_*channel_*height_*width_;
	float* weight = (float*)malloc(sizeof(float) * weight_count);
	fread(weight, sizeof(float), weight_count, fp);
	norm.weights_.reshape(num_, channel_, height_, width_);
	norm.weights_.set_data(weight);

	free(weight);
}

void readToPriorBoxLayer(PriorBoxLayer &priorbox, FILE*fp)
{
	fread(&priorbox.prior_box_param.min_size_size_, sizeof(int), 1, fp);
	for (int i = 0; i < priorbox.prior_box_param.min_size_size_; i++)
	{
		float tmp_data = 0;
		fread(&tmp_data, sizeof(float), 1, fp);
		priorbox.prior_box_param.min_size_.push_back(tmp_data);
	}
	fread(&priorbox.prior_box_param.max_size_size_, sizeof(int), 1, fp);
	for (int i = 0; i < priorbox.prior_box_param.max_size_size_; i++)
	{
		float tmp_data = 0;
		fread(&tmp_data, sizeof(float), 1, fp);
		priorbox.prior_box_param.max_size_.push_back(tmp_data);
	}

	fread(&priorbox.prior_box_param.flip_, sizeof(bool), 1, fp);
	fread(&priorbox.prior_box_param.clip_, sizeof(bool), 1, fp);

	fread(&priorbox.prior_box_param.aspact_ratio_size_, sizeof(int), 1, fp);
	for (int i = 0; i < priorbox.prior_box_param.aspact_ratio_size_; i++)
	{
		float tmp_data = 0;
		fread(&tmp_data, sizeof(float), 1, fp);
		priorbox.prior_box_param.aspact_ratio_.push_back(tmp_data);
	}
	fread(&priorbox.prior_box_param.variance_size_, sizeof(int), 1, fp);
	for (int i = 0; i < priorbox.prior_box_param.variance_size_; i++)
	{
		float tmp_data = 0;
		fread(&tmp_data, sizeof(float), 1, fp);
		priorbox.prior_box_param.variance_.push_back(tmp_data);
	}

	fread(&priorbox.prior_box_param.has_step_, sizeof(bool), 1, fp);
	fread(&priorbox.prior_box_param.step_, sizeof(float), 1, fp);
	fread(&priorbox.prior_box_param.offset_, sizeof(float), 1, fp);
}

void readToConcatLayer(ConcatLayer &concat, int axis)
{
	concat.concat_param_.concat_axis_ = axis;
}