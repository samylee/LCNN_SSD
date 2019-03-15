#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "lcnn_param.h"

class InputLayer
{
public:
	void layer_set_up(vector<Blob>& top_data);
	void forward_cpu(Mat &inputImg, vector<Blob>& top_data);

public:
	InputParam input_params_;
};

#endif