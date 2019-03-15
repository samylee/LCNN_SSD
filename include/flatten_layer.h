#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "lcnn_param.h"

class FlattenLayer
{
public:
	FlattenLayer();
	~FlattenLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
};

#endif // !FLATTEN_LAYER_H

