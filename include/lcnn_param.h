#ifndef LCNN_PARAM_H
#define LCNN_PARAM_H

#include "blob.h"

typedef struct
{
	int channel_;
	int height_;
	int width_;
}InputParam;

typedef struct
{
	int pad_size_;
	vector<int> m_pad_;
	int stride_size_;
	vector<int> m_stride_;
	int kernel_size_size_;
	vector<int> m_kernel_size_;
	int dilation_size_;
	vector<int> m_dilation_;

	bool has_pad_h_;
	bool has_pad_w_;
	int pad_h_;
	int pad_w_;

	bool has_kernel_h_;
	bool has_kernel_w_;
	int kernel_h_;
	int kernel_w_;

	bool has_stride_h_;
	bool has_stride_w_;
	int stride_h_;
	int stride_w_;

	int group_;
	int axis_;
	int num_output_;

	bool force_nd_im2col_;

	int weight_num_;
	int weight_channels_;
	int weight_height_;
	int weight_width_;
	int bias_num_;
	int bias_channels_;
	int bias_height_;
	int bias_width_;
}ConvParam;

typedef struct
{
	bool has_kernel_size_;
	int kernel_size_;
	int kernel_h_;
	int kernel_w_;

	bool has_stride_h_;
	int stride_h_;
	int stride_w_;
	int stride_;

	bool has_pad_h_;
	int pad_h_;
	int pad_w_;
	int pad_;

	bool global_pooling_;

	int pool_method_;//  MAX = 0; AVE = 1;
}PoolingParam;

typedef struct
{
	int min_size_size_;
	vector<float> min_size_;

	int max_size_size_;
	vector<float> max_size_;

	bool flip_;
	bool clip_;

	int aspact_ratio_size_;
	vector<float> aspact_ratio_;

	int variance_size_;
	vector<float> variance_;

	bool has_step_;
	float step_;
	float offset_;
}PriorBoxParam;

typedef struct
{
	int concat_axis_;
}ConcatParam;

#endif