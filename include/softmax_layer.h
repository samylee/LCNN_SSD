#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "lcnn_param.h"

class SoftmaxLayer
{
public:
	SoftmaxLayer();
	~SoftmaxLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	int outer_num_;
	int inner_num_;
	int softmax_axis_;
	/// sum_multiplier is used to carry out sum using BLAS
	Blob sum_multiplier_;
	/// scale is an intermediate Blob to hold temporary results.
	Blob scale_;
};

#endif // !SOFTMAX_LAYER_H
