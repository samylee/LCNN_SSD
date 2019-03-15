#include "conv_layer.h"

ConvolutionLayer::ConvolutionLayer()
{

}

ConvolutionLayer::~ConvolutionLayer()
{
	this->bias_.release();
	this->bias_multiplier_.release();
	this->weights_.release();
	this->col_buffer_.release();
}

void ConvolutionLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	//读取卷积核维度，一般为2维
	int channel_axis_ = this->convolution_param_.axis_;// bottom_data_[0].shape();
	int first_spatial_axis = channel_axis_ + 1;
	num_spatial_axes_ = 4 - first_spatial_axis;

	//输入nchw中的n
	this->num_ = bottom_data_[0].count(0, channel_axis_);

	//读取参数kernel
	kernel_shape_data.clear();
	if (this->convolution_param_.has_kernel_h_ || this->convolution_param_.has_kernel_w_)
	{
		kernel_shape_data.push_back(this->convolution_param_.kernel_h_);
		kernel_shape_data.push_back(this->convolution_param_.kernel_w_);
	}
	else
	{
		int num_kernel_dims = this->convolution_param_.kernel_size_size_;
		for (int i = 0; i < num_spatial_axes_; ++i)
		{
			kernel_shape_data.push_back(this->convolution_param_.m_kernel_size_[(num_kernel_dims == 1) ? 0 : i]);
		}
	}

	//读取参数stride
	stride_data.clear();
	if (this->convolution_param_.has_stride_h_ || this->convolution_param_.has_stride_w_)
	{
		stride_data.push_back(this->convolution_param_.stride_h_);
		stride_data.push_back(this->convolution_param_.stride_w_);
	}
	else
	{
		int num_stride_dims = this->convolution_param_.stride_size_;
		const int kDefaultStride = 1;
		for (int i = 0; i < num_spatial_axes_; ++i)
		{
			stride_data.push_back((num_stride_dims == 0) ? kDefaultStride : this->convolution_param_.m_stride_[(num_stride_dims == 1) ? 0 : i]);
		}
	}

	//读取参数pad
	pad_data.clear();
	if (this->convolution_param_.has_pad_h_ || this->convolution_param_.has_pad_w_)
	{
		stride_data.push_back(this->convolution_param_.pad_h_);
		stride_data.push_back(this->convolution_param_.pad_w_);
	}
	else
	{
		int num_pad_dims = this->convolution_param_.pad_size_;
		const int kDefaultPad = 0;
		for (int i = 0; i < num_spatial_axes_; ++i)
		{
			pad_data.push_back((num_pad_dims == 0) ? kDefaultPad : this->convolution_param_.m_pad_[(num_pad_dims == 1) ? 0 : i]);
		}
	}

	//读取参数dilation（膨胀？是否deconv，1表示不膨胀）
	dilation_data.clear();
	int kDefaultDilation = 1;
	int num_dilation_dims = this->convolution_param_.dilation_size_;
	for (int i = 0; i < num_spatial_axes_; ++i)
	{
		dilation_data.push_back((num_dilation_dims == 0) ? kDefaultDilation : this->convolution_param_.m_dilation_[(num_dilation_dims == 1) ? 0 : i]);
	}

	//计算输出hw
	vector<int> output_shape_;
	output_shape_.clear();
	for (int i = 0; i < num_spatial_axes_; ++i)
	{
		// i + 1 to skip channel axis
		const int input_dim = bottom_data_[0].shape(i + 1 + channel_axis_);
		const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
		const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
		output_shape_.push_back(output_dim);
	}

	//计算输出nc
	vector <int> tmp_bottom_shape_;
	tmp_bottom_shape_.clear();
	tmp_bottom_shape_.push_back(bottom_data_[0].num_);
	tmp_bottom_shape_.push_back(bottom_data_[0].channel_);
	tmp_bottom_shape_.push_back(bottom_data_[0].height_);
	tmp_bottom_shape_.push_back(bottom_data_[0].width_);
	vector<int> top_shape(tmp_bottom_shape_.begin(), tmp_bottom_shape_.begin() + channel_axis_);
	top_shape.push_back(this->convolution_param_.num_output_);

	//链接输出nchw
	for (int i = 0; i < num_spatial_axes_; ++i) {
		top_shape.push_back(output_shape_[i]);
	}

	//定义输出data
	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);

	//输入nxcxhxw
	this->bottom_dim_ = bottom_data_[0].count(channel_axis_, 4);

	//输出nxcxhxw
	this->top_dim_ = top_data_[0].count(channel_axis_, 4);

	//转换data结构以供cblas调用
	is_1x1_ = true;
	for (int i = 0; i < num_spatial_axes_; ++i)
	{
		is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
		if (!is_1x1_)
		{
			break;
		}
	}
	kernel_dim_ = this->weights_.count(1, 4);
	col_buffer_shape_.clear();
	col_buffer_shape_.push_back(kernel_dim_ * this->convolution_param_.group_);
	for (int i = 0; i < num_spatial_axes_; ++i)
	{
		col_buffer_shape_.push_back(output_shape_[i]);
	}
	while (col_buffer_shape_.size() < 4)
	{
		col_buffer_shape_.push_back(1);
	}
	col_buffer_.reshape(col_buffer_shape_[0], col_buffer_shape_[1], col_buffer_shape_[2], col_buffer_shape_[3]);
	channels_ = bottom_data_[0].shape(channel_axis_);
	conv_out_channels_ = this->convolution_param_.num_output_;
	conv_in_channels_ = channels_;
	conv_input_shape_.clear();
	for (int i = 0; i < num_spatial_axes_ + 1; ++i)
	{
		conv_input_shape_.push_back(bottom_data_[0].shape(channel_axis_ + i));
	}
	conv_out_spatial_dim_ = top_data_[0].count(first_spatial_axis, 4);
	weight_offset_ = conv_out_channels_ * kernel_dim_ / this->convolution_param_.group_;
	col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
	output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / this->convolution_param_.group_;
	conv_out_spatial_dim_ = top_data_[0].count(first_spatial_axis, 4);
	out_spatial_dim_ = top_data_[0].count(first_spatial_axis, 4);
	vector<int> bias_multiplier_shape(1, out_spatial_dim_);
	while (bias_multiplier_shape.size() < 4)
	{
		bias_multiplier_shape.push_back(1);
	}
	bias_multiplier_.reshape(bias_multiplier_shape[0], bias_multiplier_shape[1], bias_multiplier_shape[2], bias_multiplier_shape[3]);
	caffe_set(bias_multiplier_.counts_, float(1), bias_multiplier_.cpu_data());
}

void ConvolutionLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const float* weight = this->weights_.cpu_data();
	const float* bottom_data = bottom_data_[0].cpu_data();
	float* top_data = top_data_[0].cpu_data();
	for (int n = 0; n < this->num_; ++n)
	{
		this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
		const float* bias = this->bias_.cpu_data();
		this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
	}
}

void ConvolutionLayer::forward_cpu_gemm(const float* input, const float* weights,
	float* output, bool skip_im2col)
{
	const float* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_cpu(input, col_buffer_.cpu_data());
		}
		col_buff = col_buffer_.cpu_data();
	}
	for (int g = 0; g < this->convolution_param_.group_; ++g) {
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
			this->convolution_param_.group_, conv_out_spatial_dim_, kernel_dim_,
			(float)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
			(float)0., output + output_offset_ * g);
	}
}

void ConvolutionLayer::forward_cpu_bias(float* output, const float* bias)
{
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, this->convolution_param_.num_output_,
		out_spatial_dim_, 1, (float)1., bias, bias_multiplier_.cpu_data(),
		(float)1., output);
}

void ConvolutionLayer::conv_im2col_cpu(const float* data, float* col_buff)
{
	if (!this->convolution_param_.force_nd_im2col_&& num_spatial_axes_ == 2)
	{
		im2col_cpu(data, conv_in_channels_,
			conv_input_shape_[1], conv_input_shape_[2],
			kernel_shape_data[0], kernel_shape_data[1],
			pad_data[0], pad_data[1],
			stride_data[0], stride_data[1],
			dilation_data[0], dilation_data[1], col_buff);
	}
}

bool ConvolutionLayer::is_a_ge_zero_and_a_lt_b(int a, int b)
{
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void ConvolutionLayer::im2col_cpu(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	float* data_col) {
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}