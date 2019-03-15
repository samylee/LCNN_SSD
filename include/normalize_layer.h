#ifndef NORMALIZE_LAYER_H
#define NORMALIZE_LAYER_H

#include "lcnn_param.h"

class NormalizeLayer
{
public:
	NormalizeLayer();
	~NormalizeLayer();

	Blob weights_;

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	Blob norm_;
	Blob sum_channel_multiplier_, sum_spatial_multiplier_;
	Blob buffer_, buffer_channel_, buffer_spatial_;
	float eps_;
};

#endif // !NORMALIZE_LAYER_H
