#ifndef PRIOR_BOX_LAYER_H
#define PRIOR_BOX_LAYER_H

#include "lcnn_param.h"

class PriorBoxLayer
{
public:
	PriorBoxLayer();
	~PriorBoxLayer();

	PriorBoxParam prior_box_param;

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	vector<float> min_sizes_;
	vector<float> max_sizes_;
	vector<float> aspect_ratios_;
	bool flip_;
	int num_priors_;
	bool clip_;
	vector<float> variance_;

	int img_w_;
	int img_h_;
	float step_w_;
	float step_h_;

	float offset_;
};

#endif // !PRIOR_BOX_LAYER_H
