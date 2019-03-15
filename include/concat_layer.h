#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "lcnn_param.h"

class ConcatLayer
{
public:
	ConcatLayer();
	~ConcatLayer();

	ConcatParam concat_param_;

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	int count_;
	int num_concats_;
	int concat_input_size_;
	int concat_axis_;
};

#endif // !CONCAT_LAYER_H
