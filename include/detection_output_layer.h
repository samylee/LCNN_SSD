#ifndef DETECTION_OUTPUT_LAYER_H
#define DETECTION_OUTPUT_LAYER_H

#include "lcnn_param.h"

typedef struct
{
	float xmin_;
	float ymin_;
	float xmax_;
	float ymax_;

	bool has_size_;
	float size_;
}NormalizedBBox;

typedef map<int, vector<NormalizedBBox> > LabelBBox;

class DetectionOutputLayer
{
public:
	DetectionOutputLayer();
	~DetectionOutputLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	void GetLocPredictions(const float* loc_data, const int num,
		const int num_preds_per_class, const int num_loc_classes,
		const bool share_location, vector<LabelBBox>* loc_preds);

	void GetConfidenceScores(const float* conf_data, const int num,
		const int num_preds_per_class, const int num_classes,
		vector<map<int, vector<float> > >* conf_preds);

	void GetPriorBBoxes(const float* prior_data, const int num_priors,
		vector<NormalizedBBox>* prior_bboxes,
		vector<vector<float> >* prior_variances);

	float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

	void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
		const vector<NormalizedBBox>& prior_bboxes,
		const vector<vector<float> >& prior_variances,
		const int num, const bool share_location,
		const int num_loc_classes, const int background_label_id,
		const string code_type, const bool variance_encoded_in_target,
		const bool clip, vector<LabelBBox>* all_decode_bboxes);

	void DecodeBBoxes(
		const vector<NormalizedBBox>& prior_bboxes,
		const vector<vector<float> >& prior_variances,
		const string code_type, const bool variance_encoded_in_target,
		const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
		vector<NormalizedBBox>* decode_bboxes);

	void DecodeBBox(
		const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
		const string code_type, const bool variance_encoded_in_target,
		const bool clip_bbox, const NormalizedBBox& bbox,
		NormalizedBBox* decode_bbox);

	void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
		const vector<float>& scores, const float score_threshold,
		const float nms_threshold, const float eta, const int top_k,
		vector<int>* indices);

	void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
		const int top_k, vector<pair<float, int> >* score_index_vec);

	void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		NormalizedBBox* intersect_bbox);

	float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		const bool normalized = true);

private:
	float objectness_score_;
	int num_classes_;
	bool share_location_;
	int num_loc_classes_;
	int background_label_id_;
	string code_type_;
	bool variance_encoded_in_target_;
	int keep_top_k_;
	float confidence_threshold_;

	int num_;
	int num_priors_;

	float nms_threshold_;
	int top_k_;
	float eta_;
};

#endif // !DETECTION_OUTPUT_LAYER_H
