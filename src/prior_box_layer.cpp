#include "prior_box_layer.h"

PriorBoxLayer::PriorBoxLayer()
{
}

PriorBoxLayer::~PriorBoxLayer()
{
}

void PriorBoxLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	for (int i = 0; i < prior_box_param.min_size_size_; ++i) {
		min_sizes_.push_back(prior_box_param.min_size_[i]);
	}
	aspect_ratios_.clear();
	aspect_ratios_.push_back(1.);
	flip_ = prior_box_param.flip_;
	for (int i = 0; i < prior_box_param.aspact_ratio_size_; ++i) {
		float ar = prior_box_param.aspact_ratio_[i];
		bool already_exist = false;
		for (int j = 0; j < aspect_ratios_.size(); ++j) {
			if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
				already_exist = true;
				break;
			}
		}
		if (!already_exist) {
			aspect_ratios_.push_back(ar);
			if (flip_) {
				aspect_ratios_.push_back(1. / ar);
			}
		}
	}
	num_priors_ = aspect_ratios_.size() * min_sizes_.size();
	if (prior_box_param.max_size_size_ > 0) {
		for (int i = 0; i < prior_box_param.max_size_size_; ++i) {
			max_sizes_.push_back(prior_box_param.max_size_[i]);
			num_priors_ += 1;
		}
	}
	clip_ = prior_box_param.clip_;
	if (prior_box_param.variance_size_ > 1) {
		for (int i = 0; i < prior_box_param.variance_size_; ++i) {
			variance_.push_back(prior_box_param.variance_[i]);
		}
	}
	else if (prior_box_param.variance_size_ == 1) {
		variance_.push_back(prior_box_param.variance_[0]);
	}
	else {
		// Set default to 0.1.
		variance_.push_back(0.1);
	}

	img_h_ = 0;
	img_w_ = 0;

	if (prior_box_param.has_step_) {
		const float step = prior_box_param.step_;
		step_h_ = step;
		step_w_ = step;
	}
	else {
		step_h_ = 0;
		step_w_ = 0;
	}

	offset_ = prior_box_param.offset_;

	const int layer_width = bottom_data_[0].width_;
	const int layer_height = bottom_data_[0].height_;
	vector<int> top_shape(3, 1);
	// Since all images in a batch has same height and width, we only need to
	// generate one set of priors which can be shared across all images.
	top_shape[0] = 1;
	// 2 channels. First channel stores the mean of each prior coordinate.
	// Second channel stores the variance of each prior coordinate.
	top_shape[1] = 2;
	top_shape[2] = layer_width * layer_height * num_priors_ * 4;

	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], 1);
}

void PriorBoxLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const int layer_width = bottom_data_[0].width_;
	const int layer_height = bottom_data_[0].height_;
	int img_width, img_height;
	if (img_h_ == 0 || img_w_ == 0) {
		img_width = bottom_data_[1].width_;
		img_height = bottom_data_[1].height_;
	}
	else {
		img_width = img_w_;
		img_height = img_h_;
	}
	float step_w, step_h;
	if (step_w_ == 0 || step_h_ == 0) {
		step_w = static_cast<float>(img_width) / layer_width;
		step_h = static_cast<float>(img_height) / layer_height;
	}
	else {
		step_w = step_w_;
		step_h = step_h_;
	}
	float* top_data = top_data_[0].cpu_data();
	int dim = layer_height * layer_width * num_priors_ * 4;
	int idx = 0;
	for (int h = 0; h < layer_height; ++h) {
		for (int w = 0; w < layer_width; ++w) {
			float center_x = (w + offset_) * step_w;
			float center_y = (h + offset_) * step_h;
			float box_width, box_height;
			for (int s = 0; s < min_sizes_.size(); ++s) {
				int min_size_ = min_sizes_[s];
				// first prior: aspect_ratio = 1, size = min_size
				box_width = box_height = min_size_;
				// xmin
				top_data[idx++] = (center_x - box_width / 2.) / img_width;
				// ymin
				top_data[idx++] = (center_y - box_height / 2.) / img_height;
				// xmax
				top_data[idx++] = (center_x + box_width / 2.) / img_width;
				// ymax
				top_data[idx++] = (center_y + box_height / 2.) / img_height;

				if (max_sizes_.size() > 0) {
					int max_size_ = max_sizes_[s];
					// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
					box_width = box_height = sqrt(min_size_ * max_size_);
					// xmin
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					// ymin
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					// xmax
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					// ymax
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}

				// rest of priors
				for (int r = 0; r < aspect_ratios_.size(); ++r) {
					float ar = aspect_ratios_[r];
					if (fabs(ar - 1.) < 1e-6) {
						continue;
					}
					box_width = min_size_ * sqrt(ar);
					box_height = min_size_ / sqrt(ar);
					// xmin
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					// ymin
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					// xmax
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					// ymax
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}
			}
		}
	}
	// clip the prior's coordidate such that it is within [0, 1]
	if (clip_) {
		for (int d = 0; d < dim; ++d) {
			top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
		}
	}

	// set the variance.
	top_data += top_data_[0].offset(0, 1);

	if (variance_.size() == 1) {
		caffe_set(dim, float(variance_[0]), top_data);
	}
	else {
		int count = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				for (int i = 0; i < num_priors_; ++i) {
					for (int j = 0; j < 4; ++j) {
						top_data[count] = variance_[j];
						++count;
					}
				}
			}
		}
	}
}