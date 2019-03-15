#include "permute_layer.h"

PermuteLayer::PermuteLayer()
{
}

PermuteLayer::~PermuteLayer()
{
	this->permute_order_.release();
	this->old_steps_.release();
	this->new_steps_.release();
}

void Permute(const int count, float* bottom_data, const bool forward,
	const float* permute_order, const float* old_steps, const float* new_steps,
	const int num_axes, float* top_data) {
	for (int i = 0; i < count; ++i) {
		int old_idx = 0;
		int idx = i;
		for (int j = 0; j < num_axes; ++j) {
			int order = (int)(permute_order[j]);
			old_idx += (idx / (int)(new_steps[j])) * (int)(old_steps[order]);
			idx %= (int)(new_steps[j]);
		}
		if (forward) {
			top_data[i] = bottom_data[old_idx];
		}
		else {
			bottom_data[old_idx] = top_data[i];
		}
	}
}

void PermuteLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	vector<int> orders{ 0,2,3,1 };
	num_axes_ = orders.size();

	// Check if we need to reorder the data or keep it.
	need_permute_ = false;
	for (int i = 0; i < num_axes_; ++i) {
		if (orders[i] != i) {
			// As long as there is one order which is different from the natural order
			// of the data, we need to permute. Otherwise, we share the data and diff.
			need_permute_ = true;
			break;
		}
	}

	vector<int> top_shape(num_axes_, 1);
	permute_order_.reshape(num_axes_, 1, 1, 1);
	old_steps_.reshape(num_axes_, 1, 1, 1);
	new_steps_.reshape(num_axes_, 1, 1, 1);
	for (int i = 0; i < num_axes_; ++i) {
		permute_order_.cpu_data()[i] = orders[i];
		top_shape[i] = bottom_data_[0].shape(orders[i]);
	}

	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);

	vector<int> top_shape_;
	for (int i = 0; i < num_axes_; ++i) {
		if (i == num_axes_ - 1) {
			old_steps_.cpu_data()[i] = 1;
		}
		else {
			old_steps_.cpu_data()[i] = bottom_data_[0].count(i + 1, num_axes_);
		}
		top_shape_.push_back(bottom_data_[0].shape(permute_order_.cpu_data()[i]));
	}
	top_data_[0].reshape(top_shape_[0], top_shape_[1], top_shape_[2], top_shape_[3]);

	for (int i = 0; i < num_axes_; ++i) {
		if (i == num_axes_ - 1) {
			new_steps_.cpu_data()[i] = 1;
		}
		else {
			new_steps_.cpu_data()[i] = top_data_[0].count(i + 1, num_axes_);
		}
	}
}
void PermuteLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	if (need_permute_) {
		float* bottom_data = bottom_data_[0].cpu_data();
		float* top_data = top_data_[0].cpu_data();
		const int top_count = top_data_[0].counts_;
		const float* permute_order = permute_order_.cpu_data();
		const float* old_steps = old_steps_.cpu_data();
		const float* new_steps = new_steps_.cpu_data();
		bool forward = true;
		Permute(top_count, bottom_data, forward, permute_order, old_steps,
			new_steps, num_axes_, top_data);
	}
	else {
		// If there is no need to permute, we share data to save memory.
		float* bottom_data = bottom_data_[0].cpu_data();
		float* top_data = top_data_[0].cpu_data();
		const int top_count = top_data_[0].counts_;
		for (int i = 0; i < top_count; i++){
			top_data[i] = bottom_data[i];
		}
	}
}