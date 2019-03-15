#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "lcnn_param.h"

class ReluLayer
{
public:
	ReluLayer();
	~ReluLayer();

	void forward_cpu(vector<Blob>& data_blob);
};

#endif