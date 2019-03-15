#ifndef LCNN_H
#define LCNN_H

#include "ssd_common.h"

class Lcnn
{
public:
	Lcnn(string model_path);
	~Lcnn();
	vector<vector<float> > detect(Mat &image);

private:
	cv::Mat mean_;
	int target_width;
	int target_height;
	int target_channel;

//网络数据结构
private:
	vector<Blob> input_datas_;

	vector<Blob> conv1_1_datas_;
	vector<Blob> conv1_2_datas_;
	vector<Blob> pool1_datas_;

	vector<Blob> conv2_1_datas_;
	vector<Blob> conv2_2_datas_;
	vector<Blob> pool2_datas_;

	vector<Blob> conv3_1_datas_;
	vector<Blob> conv3_2_datas_;
	vector<Blob> conv3_3_datas_;
	vector<Blob> pool3_datas_;

	vector<Blob> conv4_1_datas_;
	vector<Blob> conv4_2_datas_;
	vector<Blob> conv4_3_datas_;
	vector<Blob> pool4_datas_;

	vector<Blob> conv5_1_datas_;
	vector<Blob> conv5_2_datas_;
	vector<Blob> conv5_3_datas_;
	vector<Blob> pool5_datas_;

	vector<Blob> fc6_datas_;
	vector<Blob> fc7_datas_;

	vector<Blob> conv6_1_datas_;
	vector<Blob> conv6_2_datas_;

	vector<Blob> conv7_1_datas_;
	vector<Blob> conv7_2_datas_;

	vector<Blob> conv8_1_datas_;
	vector<Blob> conv8_2_datas_;

	vector<Blob> conv9_1_datas_;
	vector<Blob> conv9_2_datas_;

	vector<Blob> conv4_3_norm_datas_;
	vector<Blob> conv4_3_norm_mbox_loc_datas_;
	vector<Blob> conv4_3_norm_mbox_loc_perm_datas_;
	vector<Blob> conv4_3_norm_mbox_loc_flat_datas_;
	vector<Blob> conv4_3_norm_mbox_conf_datas_;
	vector<Blob> conv4_3_norm_mbox_conf_perm_datas_;
	vector<Blob> conv4_3_norm_mbox_conf_flat_datas_;
	vector<Blob> conv4_3_priorbox_inputs;
	vector<Blob> conv4_3_norm_mbox_priorbox_datas_;

	vector<Blob> fc7_mbox_loc_datas_;
	vector<Blob> fc7_mbox_loc_perm_datas_;
	vector<Blob> fc7_mbox_loc_flat_datas_;
	vector<Blob> fc7_mbox_conf_datas_;
	vector<Blob> fc7_mbox_conf_perm_datas_;
	vector<Blob> fc7_mbox_conf_flat_datas_;
	vector<Blob> fc7_priorbox_inputs;
	vector<Blob> fc7_mbox_priorbox_datas_;

	vector<Blob> conv6_2_mbox_loc_datas_;
	vector<Blob> conv6_2_mbox_loc_perm_datas_;
	vector<Blob> conv6_2_mbox_loc_flat_datas_;
	vector<Blob> conv6_2_mbox_conf_datas_;
	vector<Blob> conv6_2_mbox_conf_perm_datas_;
	vector<Blob> conv6_2_mbox_conf_flat_datas_;
	vector<Blob> conv6_2_priorbox_inputs;
	vector<Blob> conv6_2_mbox_priorbox_datas_;

	vector<Blob> conv7_2_mbox_loc_datas_;
	vector<Blob> conv7_2_mbox_loc_perm_datas_;
	vector<Blob> conv7_2_mbox_loc_flat_datas_;
	vector<Blob> conv7_2_mbox_conf_datas_;
	vector<Blob> conv7_2_mbox_conf_perm_datas_;
	vector<Blob> conv7_2_mbox_conf_flat_datas_;
	vector<Blob> conv7_2_priorbox_inputs;
	vector<Blob> conv7_2_mbox_priorbox_datas_;

	vector<Blob> conv8_2_mbox_loc_datas_;
	vector<Blob> conv8_2_mbox_loc_perm_datas_;
	vector<Blob> conv8_2_mbox_loc_flat_datas_;
	vector<Blob> conv8_2_mbox_conf_datas_;
	vector<Blob> conv8_2_mbox_conf_perm_datas_;
	vector<Blob> conv8_2_mbox_conf_flat_datas_;
	vector<Blob> conv8_2_priorbox_inputs;
	vector<Blob> conv8_2_mbox_priorbox_datas_;

	vector<Blob> conv9_2_mbox_loc_datas_;
	vector<Blob> conv9_2_mbox_loc_perm_datas_;
	vector<Blob> conv9_2_mbox_loc_flat_datas_;
	vector<Blob> conv9_2_mbox_conf_datas_;
	vector<Blob> conv9_2_mbox_conf_perm_datas_;
	vector<Blob> conv9_2_mbox_conf_flat_datas_;
	vector<Blob> conv9_2_priorbox_inputs;
	vector<Blob> conv9_2_mbox_priorbox_datas_;

	vector<Blob> mbox_loc_inputs;
	vector<Blob> mbox_loc_datas_;

	vector<Blob> mbox_conf_inputs;
	vector<Blob> mbox_conf_datas_;

	vector<Blob> mbox_priorbox_inputs;
	vector<Blob> mbox_priorbox_datas_;

	vector<Blob> mbox_conf_reshape_datas_;
	vector<Blob> mbox_conf_softmax_datas_;
	vector<Blob> mbox_conf_flatten_datas_;

	vector<Blob> detection_out_inputs;
	vector<Blob> detection_out_datas_;

//网络架构
private:
	InputLayer input;

	ConvolutionLayer conv1_1;
	ReluLayer relu1_1;
	ConvolutionLayer conv1_2;
	ReluLayer relu1_2;
	PoolingLayer pool1;

	ConvolutionLayer conv2_1;
	ReluLayer relu2_1;
	ConvolutionLayer conv2_2;
	ReluLayer relu2_2;
	PoolingLayer pool2;

	ConvolutionLayer conv3_1;
	ReluLayer relu3_1;
	ConvolutionLayer conv3_2;
	ReluLayer relu3_2;
	ConvolutionLayer conv3_3;
	ReluLayer relu3_3;
	PoolingLayer pool3;

	ConvolutionLayer conv4_1;
	ReluLayer relu4_1;
	ConvolutionLayer conv4_2;
	ReluLayer relu4_2;
	ConvolutionLayer conv4_3;
	ReluLayer relu4_3;
	PoolingLayer pool4;

	ConvolutionLayer conv5_1;
	ReluLayer relu5_1;
	ConvolutionLayer conv5_2;
	ReluLayer relu5_2;
	ConvolutionLayer conv5_3;
	ReluLayer relu5_3;
	PoolingLayer pool5;

	ConvolutionLayer fc6;
	ReluLayer relu6;
	ConvolutionLayer fc7;
	ReluLayer relu7;

	ConvolutionLayer conv6_1;
	ReluLayer relu6_1;
	ConvolutionLayer conv6_2;
	ReluLayer relu6_2;

	ConvolutionLayer conv7_1;
	ReluLayer relu7_1;
	ConvolutionLayer conv7_2;
	ReluLayer relu7_2;

	ConvolutionLayer conv8_1;
	ReluLayer relu8_1;
	ConvolutionLayer conv8_2;
	ReluLayer relu8_2;

	ConvolutionLayer conv9_1;
	ReluLayer relu9_1;
	ConvolutionLayer conv9_2;
	ReluLayer relu9_2;

	NormalizeLayer conv4_3_norm;
	ConvolutionLayer conv4_3_norm_mbox_loc;
	PermuteLayer conv4_3_norm_mbox_loc_perm;
	FlattenLayer conv4_3_norm_mbox_loc_flat;
	ConvolutionLayer conv4_3_norm_mbox_conf;
	PermuteLayer conv4_3_norm_mbox_conf_perm;
	FlattenLayer conv4_3_norm_mbox_conf_flat;
	PriorBoxLayer conv4_3_norm_mbox_priorbox;

	ConvolutionLayer fc7_mbox_loc;
	PermuteLayer fc7_mbox_loc_perm;
	FlattenLayer fc7_mbox_loc_flat;
	ConvolutionLayer fc7_mbox_conf;
	PermuteLayer fc7_mbox_conf_perm;
	FlattenLayer fc7_mbox_conf_flat;
	PriorBoxLayer fc7_mbox_priorbox;

	ConvolutionLayer conv6_2_mbox_loc;
	PermuteLayer conv6_2_mbox_loc_perm;
	FlattenLayer conv6_2_mbox_loc_flat;
	ConvolutionLayer conv6_2_mbox_conf;
	PermuteLayer conv6_2_mbox_conf_perm;
	FlattenLayer conv6_2_mbox_conf_flat;
	PriorBoxLayer conv6_2_mbox_priorbox;

	ConvolutionLayer conv7_2_mbox_loc;
	PermuteLayer conv7_2_mbox_loc_perm;
	FlattenLayer conv7_2_mbox_loc_flat;
	ConvolutionLayer conv7_2_mbox_conf;
	PermuteLayer conv7_2_mbox_conf_perm;
	FlattenLayer conv7_2_mbox_conf_flat;
	PriorBoxLayer conv7_2_mbox_priorbox;

	ConvolutionLayer conv8_2_mbox_loc;
	PermuteLayer conv8_2_mbox_loc_perm;
	FlattenLayer conv8_2_mbox_loc_flat;
	ConvolutionLayer conv8_2_mbox_conf;
	PermuteLayer conv8_2_mbox_conf_perm;
	FlattenLayer conv8_2_mbox_conf_flat;
	PriorBoxLayer conv8_2_mbox_priorbox;

	ConvolutionLayer conv9_2_mbox_loc;
	PermuteLayer conv9_2_mbox_loc_perm;
	FlattenLayer conv9_2_mbox_loc_flat;
	ConvolutionLayer conv9_2_mbox_conf;
	PermuteLayer conv9_2_mbox_conf_perm;
	FlattenLayer conv9_2_mbox_conf_flat;
	PriorBoxLayer conv9_2_mbox_priorbox;

	ConcatLayer mbox_loc;
	ConcatLayer mbox_conf;
	ConcatLayer mbox_priorbox;

	ReshapeLayer mbox_conf_reshape;
	SoftmaxLayer mbox_conf_softmax;
	FlattenLayer mbox_conf_flatten;

	DetectionOutputLayer detection_out;

private:
	void setMean();
	Blob Forward(Mat &inImg);
};


#endif