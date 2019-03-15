#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "lcnn_param.h"

class ConvolutionLayer{
public:
	ConvolutionLayer();
	~ConvolutionLayer();

	ConvParam convolution_param_;

	Blob weights_;
	Blob bias_;

	int num_;
	int bottom_dim_;
	int top_dim_;

	bool is_1x1_;
	int kernel_dim_;
	vector<int> col_buffer_shape_;
	Blob col_buffer_;
	int num_spatial_axes_;
	int channels_;
	int conv_out_channels_;
	int conv_in_channels_;
	int conv_out_spatial_dim_;
	int weight_offset_;
	int col_offset_;
	int output_offset_;
	int out_spatial_dim_;

	Blob bias_multiplier_;
	vector<int> conv_input_shape_;

	vector<int> kernel_shape_data;
	vector<int> stride_data;
	vector<int> pad_data;
	vector<int> dilation_data;

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

	void forward_cpu_gemm(const float* input, const float* weights,
		float* output, bool skip_im2col = false);
	void forward_cpu_bias(float* output, const float* bias);

	void conv_im2col_cpu(const float* data, float* col_buff);

	bool is_a_ge_zero_and_a_lt_b(int a, int b);

	void im2col_cpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_col);
};

#endif