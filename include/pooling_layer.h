#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "lcnn_param.h"

class PoolingLayer
{
public:
	PoolingLayer();
	~PoolingLayer();

	PoolingParam  pool_param_;

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int pad_h_, pad_w_;
	int channels_;
	int height_, width_;
	int pooled_height_, pooled_width_;
	bool global_pooling_;
};

#endif
