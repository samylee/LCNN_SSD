#ifndef PERMUTE_LAYER_H
#define PERMUTE_LAYER_H

#include "lcnn_param.h"

class PermuteLayer
{
public:
	PermuteLayer();
	~PermuteLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	int num_axes_;
	bool need_permute_;

	Blob permute_order_;
	Blob old_steps_;
	Blob new_steps_;
};

#endif // !PERMUTE_LAYER_H

