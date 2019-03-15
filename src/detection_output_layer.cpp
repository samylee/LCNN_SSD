#include "detection_output_layer.h"

DetectionOutputLayer::DetectionOutputLayer()
{
}

DetectionOutputLayer::~DetectionOutputLayer()
{
}

void DetectionOutputLayer::layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	objectness_score_ = 0.01;
	num_classes_ = 21;
	share_location_ = true;
	num_loc_classes_ = share_location_ ? 1 : num_classes_;
	background_label_id_ = 0;
	code_type_ = "PriorBoxParameter_CodeType_CENTER_SIZE";
	variance_encoded_in_target_ = false;
	keep_top_k_ = 200;
	confidence_threshold_ = 0.01;
	// Parameters used in nms.
	nms_threshold_ = 0.45;
	eta_ = 1.0;
	top_k_ = 400;

	num_priors_ = bottom_data_[2].height_ / 4;
	// num() and channels() are 1.
	vector<int> top_shape(2, 1);
	// Since the number of bboxes to be kept is unknown before nms, we manually
	// set it to (fake) 1.
	top_shape.push_back(1);
	// Each row is a 7 dimension vector, which stores
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	top_shape.push_back(7);

	top_data_.resize(1);
	top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);

}

void DetectionOutputLayer::GetLocPredictions(const float* loc_data, const int num,
	const int num_preds_per_class, const int num_loc_classes,
	const bool share_location, vector<LabelBBox>* loc_preds) {
	loc_preds->clear();
	loc_preds->resize(num);
	for (int i = 0; i < num; ++i) {
		LabelBBox& label_bbox = (*loc_preds)[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_loc_classes * 4;
			for (int c = 0; c < num_loc_classes; ++c) {
				int label = share_location ? -1 : c;
				if (label_bbox.find(label) == label_bbox.end()) {
					label_bbox[label].resize(num_preds_per_class);
				}
				label_bbox[label][p].xmin_ = loc_data[start_idx + c * 4];
				label_bbox[label][p].ymin_ = loc_data[start_idx + c * 4 + 1];
				label_bbox[label][p].xmax_ = loc_data[start_idx + c * 4 + 2];
				label_bbox[label][p].ymax_ = loc_data[start_idx + c * 4 + 3];
				label_bbox[label][p].has_size_ = false;
			}
		}
		loc_data += num_preds_per_class * num_loc_classes * 4;
	}
}

void DetectionOutputLayer::GetConfidenceScores(const float* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	vector<map<int, vector<float> > >* conf_preds)
{
	conf_preds->clear();
	conf_preds->resize(num);
	for (int i = 0; i < num; ++i) {
		map<int, vector<float> >& label_scores = (*conf_preds)[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_classes;
			for (int c = 0; c < num_classes; ++c) {
				label_scores[c].push_back(conf_data[start_idx + c]);
			}
		}
		conf_data += num_preds_per_class * num_classes;
	}
}

float DetectionOutputLayer::BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
	if (bbox.xmax_ < bbox.xmin_ || bbox.ymax_ < bbox.ymin_) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	}
	else {
		if (bbox.has_size_) {
			return bbox.size_;
		}
		else {
			float width = bbox.xmax_ - bbox.xmin_;
			float height = bbox.ymax_ - bbox.ymin_;
			if (normalized) {
				return width * height;
			}
			else {
				// If bbox is not within range [0, 1].
				return (width + 1) * (height + 1);
			}
		}
	}
}

void DetectionOutputLayer::GetPriorBBoxes(const float* prior_data, const int num_priors,
	vector<NormalizedBBox>* prior_bboxes,
	vector<vector<float> >* prior_variances) {
	prior_bboxes->clear();
	prior_variances->clear();
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = i * 4;
		NormalizedBBox bbox;
		bbox.xmin_ = prior_data[start_idx];
		bbox.ymin_ = prior_data[start_idx + 1];
		bbox.xmax_ = prior_data[start_idx + 2];
		bbox.ymax_ = prior_data[start_idx + 3];
		bbox.has_size_ = false;
		float bbox_size = BBoxSize(bbox);
		bbox.size_ = bbox_size;
		bbox.has_size_ = true;
		prior_bboxes->push_back(bbox);
	}

	for (int i = 0; i < num_priors; ++i) {
		int start_idx = (num_priors + i) * 4;
		vector<float> var;
		for (int j = 0; j < 4; ++j) {
			var.push_back(prior_data[start_idx + j]);
		}
		prior_variances->push_back(var);
	}
}

void DetectionOutputLayer::DecodeBBoxes(
	const vector<NormalizedBBox>& prior_bboxes,
	const vector<vector<float> >& prior_variances,
	const string code_type, const bool variance_encoded_in_target,
	const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
	vector<NormalizedBBox>* decode_bboxes) {
	int num_bboxes = prior_bboxes.size();
	decode_bboxes->clear();
	for (int i = 0; i < num_bboxes; ++i) {
		NormalizedBBox decode_bbox;
		DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
			variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
		decode_bboxes->push_back(decode_bbox);
	}
}

void DetectionOutputLayer::DecodeBBox(
	const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
	const string code_type, const bool variance_encoded_in_target,
	const bool clip_bbox, const NormalizedBBox& bbox,
	NormalizedBBox* decode_bbox)
{
	if (code_type == "PriorBoxParameter_CodeType_CENTER_SIZE") {
		float prior_width = prior_bbox.xmax_ - prior_bbox.xmin_;
		float prior_height = prior_bbox.ymax_ - prior_bbox.ymin_;
		float prior_center_x = (prior_bbox.xmin_ + prior_bbox.xmax_) / 2.;
		float prior_center_y = (prior_bbox.ymin_ + prior_bbox.ymax_) / 2.;

		float decode_bbox_center_x, decode_bbox_center_y;
		float decode_bbox_width, decode_bbox_height;
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to retore the offset
			// predictions.
			decode_bbox_center_x = bbox.xmin_ * prior_width + prior_center_x;
			decode_bbox_center_y = bbox.ymin_ * prior_height + prior_center_y;
			decode_bbox_width = exp(bbox.xmax_) * prior_width;
			decode_bbox_height = exp(bbox.ymax_) * prior_height;
		}
		else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox_center_x =
				prior_variance[0] * bbox.xmin_ * prior_width + prior_center_x;
			decode_bbox_center_y =
				prior_variance[1] * bbox.ymin_ * prior_height + prior_center_y;
			decode_bbox_width =
				exp(prior_variance[2] * bbox.xmax_) * prior_width;
			decode_bbox_height =
				exp(prior_variance[3] * bbox.ymax_) * prior_height;
		}

		decode_bbox->xmin_ = decode_bbox_center_x - decode_bbox_width / 2.;
		decode_bbox->ymin_ = decode_bbox_center_y - decode_bbox_height / 2.;
		decode_bbox->xmax_ = decode_bbox_center_x + decode_bbox_width / 2.;
		decode_bbox->ymax_ = decode_bbox_center_y + decode_bbox_height / 2.;
	}
	else {
		cout << "Unknown LocLossType.";
	}
	decode_bbox->has_size_ = false;
	float bbox_size = BBoxSize(*decode_bbox);
	decode_bbox->size_ = bbox_size;
	decode_bbox->has_size_ = true;
}

void DetectionOutputLayer::DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
	const vector<NormalizedBBox>& prior_bboxes,
	const vector<vector<float> >& prior_variances,
	const int num, const bool share_location,
	const int num_loc_classes, const int background_label_id,
	const string code_type, const bool variance_encoded_in_target,
	const bool clip, vector<LabelBBox>* all_decode_bboxes) {
	all_decode_bboxes->clear();
	all_decode_bboxes->resize(num);
	for (int i = 0; i < num; ++i) {
		// Decode predictions into bboxes.
		LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
		for (int c = 0; c < num_loc_classes; ++c) {
			int label = share_location ? -1 : c;
			if (label == background_label_id) {
				// Ignore background class.
				continue;
			}
			const vector<NormalizedBBox>& label_loc_preds =
				all_loc_preds[i].find(label)->second;
			DecodeBBoxes(prior_bboxes, prior_variances,
				code_type, variance_encoded_in_target, clip,
				label_loc_preds, &(decode_bboxes[label]));
		}
	}
}

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
	const pair<float, T>& pair2) {
	return pair1.first > pair2.first;
}

void DetectionOutputLayer::GetMaxScoreIndex(const vector<float>& scores, const float threshold,
	const int top_k, vector<pair<float, int> >* score_index_vec) {
	// Generate index score pairs.
	for (int i = 0; i < scores.size(); ++i) {
		if (scores[i] > threshold) {
			score_index_vec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
		SortScorePairDescend<int>);

	// Keep top_k scores if needed.
	if (top_k > -1 && top_k < score_index_vec->size()) {
		score_index_vec->resize(top_k);
	}
}

void DetectionOutputLayer::IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
	NormalizedBBox* intersect_bbox) {
	if (bbox2.xmin_ > bbox1.xmax_ || bbox2.xmax_ < bbox1.xmin_ ||
		bbox2.ymin_ > bbox1.ymax_ || bbox2.ymax_ < bbox1.ymin_) {
		// Return [0, 0, 0, 0] if there is no intersection.
		intersect_bbox->xmin_ = 0;
		intersect_bbox->ymin_ = 0;
		intersect_bbox->xmax_ = 0;
		intersect_bbox->ymax_ = 0;
	}
	else {
		intersect_bbox->xmin_ = (std::max(bbox1.xmin_, bbox2.xmin_));
		intersect_bbox->ymin_ = (std::max(bbox1.ymin_, bbox2.ymin_));
		intersect_bbox->xmax_ = (std::min(bbox1.xmax_, bbox2.xmax_));
		intersect_bbox->ymax_ = (std::min(bbox1.ymax_, bbox2.ymax_));
	}
}

float DetectionOutputLayer::JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
	const bool normalized) {
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_width, intersect_height;
	if (normalized) {
		intersect_width = intersect_bbox.xmax_ - intersect_bbox.xmin_;
		intersect_height = intersect_bbox.ymax_ - intersect_bbox.ymin_;
	}
	else {
		intersect_width = intersect_bbox.xmax_ - intersect_bbox.xmin_ + 1;
		intersect_height = intersect_bbox.ymax_ - intersect_bbox.ymin_ + 1;
	}
	if (intersect_width > 0 && intersect_height > 0) {
		float intersect_size = intersect_width * intersect_height;
		float bbox1_size = BBoxSize(bbox1);
		float bbox2_size = BBoxSize(bbox2);
		return intersect_size / (bbox1_size + bbox2_size - intersect_size);
	}
	else {
		return 0.;
	}
}

void DetectionOutputLayer::ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
	const vector<float>& scores, const float score_threshold,
	const float nms_threshold, const float eta, const int top_k,
	vector<int>* indices) {
	// Get top_k scores (with corresponding indices).
	vector<pair<float, int> > score_index_vec;
	GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

	// Do nms.
	float adaptive_threshold = nms_threshold;
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
				keep = overlap <= adaptive_threshold;
			}
			else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}

void DetectionOutputLayer::forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_)
{
	const float* loc_data = bottom_data_[0].cpu_data();
	const float* conf_data = bottom_data_[1].cpu_data();
	const float* prior_data = bottom_data_[2].cpu_data();

	const int num = bottom_data_[0].num_;
	// Retrieve all location predictions.
	vector<LabelBBox> all_loc_preds;
	GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
		share_location_, &all_loc_preds);

	// Retrieve all confidences.
	vector<map<int, vector<float> > > all_conf_scores;
	GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
			&all_conf_scores);

	// Retrieve all prior bboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

	// Decode all loc predictions to bboxes.
	vector<LabelBBox> all_decode_bboxes;
	const bool clip_bbox = false;
	DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
			share_location_, num_loc_classes_, background_label_id_,
			code_type_, variance_encoded_in_target_, clip_bbox,
			&all_decode_bboxes);

	int num_kept = 0;
	vector<map<int, vector<int> > > all_indices;
	for (int i = 0; i < num; ++i) {
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		map<int, vector<int> > indices;
		int num_det = 0;
		for (int c = 0; c < num_classes_; ++c) {
			if (c == background_label_id_) {
				// Ignore background class.
				continue;
			}
			const vector<float>& scores = conf_scores.find(c)->second;
			int label = share_location_ ? -1 : c;
			if (decode_bboxes.find(label) == decode_bboxes.end()) {
				continue;
			}
			const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
				top_k_, &(indices[c]));
			num_det += indices[c].size();
		}
		if (keep_top_k_ > -1 && num_det > keep_top_k_) {
			vector<pair<float, pair<int, int> > > score_index_pairs;
			for (map<int, vector<int> >::iterator it = indices.begin();
				it != indices.end(); ++it) {
				int label = it->first;
				const vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end()) {
					continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				for (int j = 0; j < label_indices.size(); ++j) {
					int idx = label_indices[j];
					score_index_pairs.push_back(std::make_pair(
						scores[idx], std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
				SortScorePairDescend<pair<int, int> >);
			score_index_pairs.resize(keep_top_k_);
			// Store the new indices.
			map<int, vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j) {
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += keep_top_k_;
		}
		else {
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(7);
	float* top_data;
	if (num_kept == 0) {
		top_shape[2] = num;
		top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
		top_data = top_data_[0].cpu_data();
		caffe_set(top_data_[0].counts_, -1, top_data);
		// Generate fake results per image.
		for (int i = 0; i < num; ++i) {
			top_data[0] = i;
			top_data += 7;
		}
	}
	else {
		top_data_[0].reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
		top_data = top_data_[0].cpu_data();
	}

	int count = 0;
	for (int i = 0; i < num; ++i) {
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		for (map<int, vector<int> >::iterator it = all_indices[i].begin();
			it != all_indices[i].end(); ++it) {
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				continue;
			}
			const vector<float>& scores = conf_scores.find(label)->second;
			int loc_label = share_location_ ? -1 : label;
			if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				continue;
			}
			const vector<NormalizedBBox>& bboxes =
				decode_bboxes.find(loc_label)->second;
			vector<int>& indices = it->second;
			for (int j = 0; j < indices.size(); ++j) {
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = scores[idx];
				const NormalizedBBox& bbox = bboxes[idx];
				top_data[count * 7 + 3] = bbox.xmin_;
				top_data[count * 7 + 4] = bbox.ymin_;
				top_data[count * 7 + 5] = bbox.xmax_;
				top_data[count * 7 + 6] = bbox.ymax_;
				++count;
			}
		}
	}

	//const float* bottom_data_tmp = bottom_data_[0].cpu_data();
	//static int count_layer = 1;
	//string dst_path = "data_file/output/" + to_string(count_layer);
	//ofstream dst_in(dst_path + "dst_in.txt");
	//for (int i = 0; i < bottom_data_[0].counts_; i++)
	//{
	//	stringstream strStream;
	//	strStream << bottom_data_tmp[i];
	//	dst_in << strStream.str() << " ";
	//	if ((i + 1) % 10 == 0)
	//		dst_in << endl;
	//}
	//dst_in.close();

	//float* top_data_tmp = top_data_[0].cpu_data();
	//ofstream dst_out(dst_path + "dst_out.txt");
	//for (int i = 0; i < top_data_[0].counts_; i++)
	//{
	//	stringstream strStream;
	//	strStream << top_data_tmp[i];
	//	dst_out << strStream.str() << " ";
	//	if ((i+1) % 10 == 0)
	//		dst_out << endl;
	//}
	//dst_out.close();
	//count_layer++;
}