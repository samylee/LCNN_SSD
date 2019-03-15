#ifndef RESHAPE_LAYER_H
#define RESHAPE_LAYER_H

#include "lcnn_param.h"

class ReshapeLayer
{
public:
	ReshapeLayer();
	~ReshapeLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
};

#endif // !RESHAPE_LAYER_H
