#include "lcnn.h"
#include <fstream>

Lcnn::Lcnn(string model_path)
{
	FILE* fp = fopen(model_path.c_str(), "rb");
	if (fp == NULL)
	{
		cout << "File could not be opened!" << endl;
		system("pause");
		exit(0);
	}
	
	///////////////////////////////network//////////////////////////////
	//input_layer
	readToInputLayer(input, fp);
	input.layer_set_up(input_datas_);

	//conv1
	readToConvLayer(conv1_1, fp);
	conv1_1.layer_set_up(input_datas_, conv1_1_datas_);
	readToConvLayer(conv1_2, fp);
	conv1_2.layer_set_up(conv1_1_datas_, conv1_2_datas_);
	readToPoolingLayer(pool1, fp);
	pool1.layer_set_up(conv1_2_datas_, pool1_datas_);

	//conv2
	readToConvLayer(conv2_1, fp);
	conv2_1.layer_set_up(pool1_datas_, conv2_1_datas_);
	readToConvLayer(conv2_2, fp);
	conv2_2.layer_set_up(conv2_1_datas_, conv2_2_datas_);
	readToPoolingLayer(pool2, fp);
	pool2.layer_set_up(conv2_2_datas_, pool2_datas_);

	//conv3
	readToConvLayer(conv3_1, fp);
	conv3_1.layer_set_up(pool2_datas_, conv3_1_datas_);
	readToConvLayer(conv3_2, fp);
	conv3_2.layer_set_up(conv3_1_datas_, conv3_2_datas_);
	readToConvLayer(conv3_3, fp);
	conv3_3.layer_set_up(conv3_2_datas_, conv3_3_datas_);
	readToPoolingLayer(pool3, fp);
	pool3.layer_set_up(conv3_3_datas_, pool3_datas_);

	//conv4
	readToConvLayer(conv4_1, fp);
	conv4_1.layer_set_up(pool3_datas_, conv4_1_datas_);
	readToConvLayer(conv4_2, fp);
	conv4_2.layer_set_up(conv4_1_datas_, conv4_2_datas_);
	readToConvLayer(conv4_3, fp);
	conv4_3.layer_set_up(conv4_2_datas_, conv4_3_datas_);
	readToPoolingLayer(pool4, fp);
	pool4.layer_set_up(conv4_3_datas_, pool4_datas_);

	//conv5
	readToConvLayer(conv5_1, fp);
	conv5_1.layer_set_up(pool4_datas_, conv5_1_datas_);
	readToConvLayer(conv5_2, fp);
	conv5_2.layer_set_up(conv5_1_datas_, conv5_2_datas_);
	readToConvLayer(conv5_3, fp);
	conv5_3.layer_set_up(conv5_2_datas_, conv5_3_datas_);
	readToPoolingLayer(pool5, fp);
	pool5.layer_set_up(conv5_3_datas_, pool5_datas_);

	//fc6
	readToConvLayer(fc6, fp);
	fc6.layer_set_up(pool5_datas_, fc6_datas_);

	//fc7
	readToConvLayer(fc7, fp);
	fc7.layer_set_up(fc6_datas_, fc7_datas_);

	//conv6
	readToConvLayer(conv6_1, fp);
	conv6_1.layer_set_up(fc7_datas_, conv6_1_datas_);
	readToConvLayer(conv6_2, fp);
	conv6_2.layer_set_up(conv6_1_datas_, conv6_2_datas_);

	//conv7
	readToConvLayer(conv7_1, fp);
	conv7_1.layer_set_up(conv6_2_datas_, conv7_1_datas_);
	readToConvLayer(conv7_2, fp);
	conv7_2.layer_set_up(conv7_1_datas_, conv7_2_datas_);

	//conv8
	readToConvLayer(conv8_1, fp);
	conv8_1.layer_set_up(conv7_2_datas_, conv8_1_datas_);
	readToConvLayer(conv8_2, fp);
	conv8_2.layer_set_up(conv8_1_datas_, conv8_2_datas_);

	//conv9
	readToConvLayer(conv9_1, fp);
	conv9_1.layer_set_up(conv8_2_datas_, conv9_1_datas_);
	readToConvLayer(conv9_2, fp);
	conv9_2.layer_set_up(conv9_1_datas_, conv9_2_datas_);

	//conv4_3_norm
	readToNormalizeLayer(conv4_3_norm, fp);
	conv4_3_norm.layer_set_up(conv4_3_datas_, conv4_3_norm_datas_);
	//conv4_3_norm_mbox_loc
	readToConvLayer(conv4_3_norm_mbox_loc, fp);
	conv4_3_norm_mbox_loc.layer_set_up(conv4_3_norm_datas_, conv4_3_norm_mbox_loc_datas_);
	//conv4_3_norm_mbox_loc_perm
	conv4_3_norm_mbox_loc_perm.layer_set_up(conv4_3_norm_mbox_loc_datas_, conv4_3_norm_mbox_loc_perm_datas_);
	//conv4_3_norm_mbox_loc_flat
	conv4_3_norm_mbox_loc_flat.layer_set_up(conv4_3_norm_mbox_loc_perm_datas_, conv4_3_norm_mbox_loc_flat_datas_);
	//conv4_3_norm_mbox_conf
	readToConvLayer(conv4_3_norm_mbox_conf, fp);
	conv4_3_norm_mbox_conf.layer_set_up(conv4_3_norm_datas_, conv4_3_norm_mbox_conf_datas_);
	//conv4_3_norm_mbox_conf_perm
	conv4_3_norm_mbox_conf_perm.layer_set_up(conv4_3_norm_mbox_conf_datas_, conv4_3_norm_mbox_conf_perm_datas_);
	//conv4_3_norm_mbox_conf_flat
	conv4_3_norm_mbox_conf_flat.layer_set_up(conv4_3_norm_mbox_conf_perm_datas_, conv4_3_norm_mbox_conf_flat_datas_);
	//conv4_3_norm_mbox_priorbox
	readToPriorBoxLayer(conv4_3_norm_mbox_priorbox, fp);
	conv4_3_priorbox_inputs.push_back(conv4_3_norm_datas_[0]);
	conv4_3_priorbox_inputs.push_back(input_datas_[0]);
	conv4_3_norm_mbox_priorbox.layer_set_up(conv4_3_priorbox_inputs, conv4_3_norm_mbox_priorbox_datas_);

	//fc7_mbox_loc
	readToConvLayer(fc7_mbox_loc, fp);
	fc7_mbox_loc.layer_set_up(fc7_datas_, fc7_mbox_loc_datas_);
	//fc7_mbox_loc_perm
	fc7_mbox_loc_perm.layer_set_up(fc7_mbox_loc_datas_, fc7_mbox_loc_perm_datas_);
	//fc7_mbox_loc_flat
	fc7_mbox_loc_flat.layer_set_up(fc7_mbox_loc_perm_datas_, fc7_mbox_loc_flat_datas_);
	//fc7_mbox_conf
	readToConvLayer(fc7_mbox_conf, fp);
	fc7_mbox_conf.layer_set_up(fc7_datas_, fc7_mbox_conf_datas_);
	//fc7_mbox_conf_perm
	fc7_mbox_conf_perm.layer_set_up(fc7_mbox_conf_datas_, fc7_mbox_conf_perm_datas_);
	//fc7_mbox_conf_flat
	fc7_mbox_conf_flat.layer_set_up(fc7_mbox_conf_perm_datas_, fc7_mbox_conf_flat_datas_);
	//fc7_mbox_priorbox
	readToPriorBoxLayer(fc7_mbox_priorbox, fp);
	fc7_priorbox_inputs.push_back(fc7_datas_[0]);
	fc7_priorbox_inputs.push_back(input_datas_[0]);
	fc7_mbox_priorbox.layer_set_up(fc7_priorbox_inputs, fc7_mbox_priorbox_datas_);

	//conv6_2_mbox_loc
	readToConvLayer(conv6_2_mbox_loc, fp);
	conv6_2_mbox_loc.layer_set_up(conv6_2_datas_, conv6_2_mbox_loc_datas_);
	//conv6_2_mbox_loc_perm
	conv6_2_mbox_loc_perm.layer_set_up(conv6_2_mbox_loc_datas_, conv6_2_mbox_loc_perm_datas_);
	//conv6_2_mbox_loc_flat
	conv6_2_mbox_loc_flat.layer_set_up(conv6_2_mbox_loc_perm_datas_, conv6_2_mbox_loc_flat_datas_);
	//conv6_2_mbox_conf
	readToConvLayer(conv6_2_mbox_conf, fp);
	conv6_2_mbox_conf.layer_set_up(conv6_2_datas_, conv6_2_mbox_conf_datas_);
	//conv6_2_mbox_conf_perm
	conv6_2_mbox_conf_perm.layer_set_up(conv6_2_mbox_conf_datas_, conv6_2_mbox_conf_perm_datas_);
	//conv6_2_mbox_conf_flat
	conv6_2_mbox_conf_flat.layer_set_up(conv6_2_mbox_conf_perm_datas_, conv6_2_mbox_conf_flat_datas_);
	//conv6_2_mbox_priorbox
	readToPriorBoxLayer(conv6_2_mbox_priorbox, fp);
	conv6_2_priorbox_inputs.push_back(conv6_2_datas_[0]);
	conv6_2_priorbox_inputs.push_back(input_datas_[0]);
	conv6_2_mbox_priorbox.layer_set_up(conv6_2_priorbox_inputs, conv6_2_mbox_priorbox_datas_);

	//conv7_2_mbox_loc
	readToConvLayer(conv7_2_mbox_loc, fp);
	conv7_2_mbox_loc.layer_set_up(conv7_2_datas_, conv7_2_mbox_loc_datas_);
	//conv7_2_mbox_loc_perm
	conv7_2_mbox_loc_perm.layer_set_up(conv7_2_mbox_loc_datas_, conv7_2_mbox_loc_perm_datas_);
	//conv7_2_mbox_loc_flat
	conv7_2_mbox_loc_flat.layer_set_up(conv7_2_mbox_loc_perm_datas_, conv7_2_mbox_loc_flat_datas_);
	//conv7_2_mbox_conf
	readToConvLayer(conv7_2_mbox_conf, fp);
	conv7_2_mbox_conf.layer_set_up(conv7_2_datas_, conv7_2_mbox_conf_datas_);
	//conv7_2_mbox_conf_perm
	conv7_2_mbox_conf_perm.layer_set_up(conv7_2_mbox_conf_datas_, conv7_2_mbox_conf_perm_datas_);
	//conv7_2_mbox_conf_flat
	conv7_2_mbox_conf_flat.layer_set_up(conv7_2_mbox_conf_perm_datas_, conv7_2_mbox_conf_flat_datas_);
	//conv7_2_mbox_priorbox
	readToPriorBoxLayer(conv7_2_mbox_priorbox, fp);
	conv7_2_priorbox_inputs.push_back(conv7_2_datas_[0]);
	conv7_2_priorbox_inputs.push_back(input_datas_[0]);
	conv7_2_mbox_priorbox.layer_set_up(conv7_2_priorbox_inputs, conv7_2_mbox_priorbox_datas_);

	//conv8_2_mbox_loc
	readToConvLayer(conv8_2_mbox_loc, fp);
	conv8_2_mbox_loc.layer_set_up(conv8_2_datas_, conv8_2_mbox_loc_datas_);
	//conv8_2_mbox_loc_perm
	conv8_2_mbox_loc_perm.layer_set_up(conv8_2_mbox_loc_datas_, conv8_2_mbox_loc_perm_datas_);
	//conv8_2_mbox_loc_flat
	conv8_2_mbox_loc_flat.layer_set_up(conv8_2_mbox_loc_perm_datas_, conv8_2_mbox_loc_flat_datas_);
	//conv8_2_mbox_conf
	readToConvLayer(conv8_2_mbox_conf, fp);
	conv8_2_mbox_conf.layer_set_up(conv8_2_datas_, conv8_2_mbox_conf_datas_);
	//conv8_2_mbox_conf_perm
	conv8_2_mbox_conf_perm.layer_set_up(conv8_2_mbox_conf_datas_, conv8_2_mbox_conf_perm_datas_);
	//conv8_2_mbox_conf_flat
	conv8_2_mbox_conf_flat.layer_set_up(conv8_2_mbox_conf_perm_datas_, conv8_2_mbox_conf_flat_datas_);
	//conv8_2_mbox_priorbox
	readToPriorBoxLayer(conv8_2_mbox_priorbox, fp);
	conv8_2_priorbox_inputs.push_back(conv8_2_datas_[0]);
	conv8_2_priorbox_inputs.push_back(input_datas_[0]);
	conv8_2_mbox_priorbox.layer_set_up(conv8_2_priorbox_inputs, conv8_2_mbox_priorbox_datas_);

	//conv9_2_mbox_loc
	readToConvLayer(conv9_2_mbox_loc, fp);
	conv9_2_mbox_loc.layer_set_up(conv9_2_datas_, conv9_2_mbox_loc_datas_);
	//conv9_2_mbox_loc_perm
	conv9_2_mbox_loc_perm.layer_set_up(conv9_2_mbox_loc_datas_, conv9_2_mbox_loc_perm_datas_);
	//conv9_2_mbox_loc_flat
	conv9_2_mbox_loc_flat.layer_set_up(conv9_2_mbox_loc_perm_datas_, conv9_2_mbox_loc_flat_datas_);
	//conv9_2_mbox_conf
	readToConvLayer(conv9_2_mbox_conf, fp);
	conv9_2_mbox_conf.layer_set_up(conv9_2_datas_, conv9_2_mbox_conf_datas_);
	//conv9_2_mbox_conf_perm
	conv9_2_mbox_conf_perm.layer_set_up(conv9_2_mbox_conf_datas_, conv9_2_mbox_conf_perm_datas_);
	//conv9_2_mbox_conf_flat
	conv9_2_mbox_conf_flat.layer_set_up(conv9_2_mbox_conf_perm_datas_, conv9_2_mbox_conf_flat_datas_);
	//conv9_2_mbox_priorbox
	readToPriorBoxLayer(conv9_2_mbox_priorbox, fp);
	conv9_2_priorbox_inputs.push_back(conv9_2_datas_[0]);
	conv9_2_priorbox_inputs.push_back(input_datas_[0]);
	conv9_2_mbox_priorbox.layer_set_up(conv9_2_priorbox_inputs, conv9_2_mbox_priorbox_datas_);

	readToConcatLayer(mbox_loc, 1);
	mbox_loc_inputs.push_back(conv4_3_norm_mbox_loc_flat_datas_[0]);
	mbox_loc_inputs.push_back(fc7_mbox_loc_flat_datas_[0]);
	mbox_loc_inputs.push_back(conv6_2_mbox_loc_flat_datas_[0]);
	mbox_loc_inputs.push_back(conv7_2_mbox_loc_flat_datas_[0]);
	mbox_loc_inputs.push_back(conv8_2_mbox_loc_flat_datas_[0]);
	mbox_loc_inputs.push_back(conv9_2_mbox_loc_flat_datas_[0]);
	mbox_loc.layer_set_up(mbox_loc_inputs, mbox_loc_datas_);

	readToConcatLayer(mbox_conf, 1);
	mbox_conf_inputs.push_back(conv4_3_norm_mbox_conf_flat_datas_[0]);
	mbox_conf_inputs.push_back(fc7_mbox_conf_flat_datas_[0]);
	mbox_conf_inputs.push_back(conv6_2_mbox_conf_flat_datas_[0]);
	mbox_conf_inputs.push_back(conv7_2_mbox_conf_flat_datas_[0]);
	mbox_conf_inputs.push_back(conv8_2_mbox_conf_flat_datas_[0]);
	mbox_conf_inputs.push_back(conv9_2_mbox_conf_flat_datas_[0]);
	mbox_conf.layer_set_up(mbox_conf_inputs, mbox_conf_datas_);

	readToConcatLayer(mbox_priorbox, 2);
	mbox_priorbox_inputs.push_back(conv4_3_norm_mbox_priorbox_datas_[0]);
	mbox_priorbox_inputs.push_back(fc7_mbox_priorbox_datas_[0]);
	mbox_priorbox_inputs.push_back(conv6_2_mbox_priorbox_datas_[0]);
	mbox_priorbox_inputs.push_back(conv7_2_mbox_priorbox_datas_[0]);
	mbox_priorbox_inputs.push_back(conv8_2_mbox_priorbox_datas_[0]);
	mbox_priorbox_inputs.push_back(conv9_2_mbox_priorbox_datas_[0]);
	mbox_priorbox.layer_set_up(mbox_priorbox_inputs, mbox_priorbox_datas_);

	mbox_conf_reshape.layer_set_up(mbox_conf_datas_, mbox_conf_reshape_datas_);
	mbox_conf_softmax.layer_set_up(mbox_conf_reshape_datas_, mbox_conf_softmax_datas_);
	mbox_conf_flatten.layer_set_up(mbox_conf_softmax_datas_, mbox_conf_flatten_datas_);

	detection_out_inputs.push_back(mbox_loc_datas_[0]);
	detection_out_inputs.push_back(mbox_conf_flatten_datas_[0]);
	detection_out_inputs.push_back(mbox_priorbox_datas_[0]);
	detection_out.layer_set_up(detection_out_inputs, detection_out_datas_);

	///////////////////////////////network//////////////////////////////

	fclose(fp);

	target_width = input.input_params_.width_;
	target_height = input.input_params_.height_;
	target_channel = input.input_params_.channel_;
	setMean();
}

Lcnn::~Lcnn()
{
	destroy_blob(this->input_datas_);

	destroy_blob(this->conv1_1_datas_);
	destroy_blob(this->conv1_2_datas_);
	destroy_blob(this->pool1_datas_);

	destroy_blob(this->conv2_1_datas_);
	destroy_blob(this->conv2_2_datas_);
	destroy_blob(this->pool2_datas_);

	destroy_blob(this->conv3_1_datas_);
	destroy_blob(this->conv3_2_datas_);
	destroy_blob(this->conv3_3_datas_);
	destroy_blob(this->pool3_datas_);

	destroy_blob(this->conv4_1_datas_);
	destroy_blob(this->conv4_2_datas_);
	destroy_blob(this->conv4_3_datas_);
	destroy_blob(this->pool4_datas_);

	destroy_blob(this->conv5_1_datas_);
	destroy_blob(this->conv5_2_datas_);
	destroy_blob(this->conv5_3_datas_);
	destroy_blob(this->pool5_datas_);

	destroy_blob(this->fc6_datas_);
	destroy_blob(this->fc7_datas_);

	destroy_blob(this->conv6_1_datas_);
	destroy_blob(this->conv6_2_datas_);

	destroy_blob(this->conv7_1_datas_);
	destroy_blob(this->conv7_2_datas_);

	destroy_blob(this->conv8_1_datas_);
	destroy_blob(this->conv8_2_datas_);

	destroy_blob(this->conv9_1_datas_);
	destroy_blob(this->conv9_2_datas_);

	destroy_blob(this->conv4_3_norm_datas_);
	destroy_blob(this->conv4_3_norm_mbox_loc_datas_);
	destroy_blob(this->conv4_3_norm_mbox_loc_perm_datas_);
	destroy_blob(this->conv4_3_norm_mbox_loc_flat_datas_);
	destroy_blob(this->conv4_3_norm_mbox_conf_datas_);
	destroy_blob(this->conv4_3_norm_mbox_conf_perm_datas_);
	destroy_blob(this->conv4_3_norm_mbox_conf_flat_datas_);
	destroy_blob(this->conv4_3_norm_mbox_priorbox_datas_);

	destroy_blob(this->fc7_mbox_loc_datas_);
	destroy_blob(this->fc7_mbox_loc_perm_datas_);
	destroy_blob(this->fc7_mbox_loc_flat_datas_);
	destroy_blob(this->fc7_mbox_conf_datas_);
	destroy_blob(this->fc7_mbox_conf_perm_datas_);
	destroy_blob(this->fc7_mbox_conf_flat_datas_);
	destroy_blob(this->fc7_mbox_priorbox_datas_);

	destroy_blob(this->conv6_2_mbox_loc_datas_);
	destroy_blob(this->conv6_2_mbox_loc_perm_datas_);
	destroy_blob(this->conv6_2_mbox_loc_flat_datas_);
	destroy_blob(this->conv6_2_mbox_conf_datas_);
	destroy_blob(this->conv6_2_mbox_conf_perm_datas_);
	destroy_blob(this->conv6_2_mbox_conf_flat_datas_);
	destroy_blob(this->conv6_2_mbox_priorbox_datas_);

	destroy_blob(this->conv7_2_mbox_loc_datas_);
	destroy_blob(this->conv7_2_mbox_loc_perm_datas_);
	destroy_blob(this->conv7_2_mbox_loc_flat_datas_);
	destroy_blob(this->conv7_2_mbox_conf_datas_);
	destroy_blob(this->conv7_2_mbox_conf_perm_datas_);
	destroy_blob(this->conv7_2_mbox_conf_flat_datas_);
	destroy_blob(this->conv7_2_mbox_priorbox_datas_);

	destroy_blob(this->conv8_2_mbox_loc_datas_);
	destroy_blob(this->conv8_2_mbox_loc_perm_datas_);
	destroy_blob(this->conv8_2_mbox_loc_flat_datas_);
	destroy_blob(this->conv8_2_mbox_conf_datas_);
	destroy_blob(this->conv8_2_mbox_conf_perm_datas_);
	destroy_blob(this->conv8_2_mbox_conf_flat_datas_);
	destroy_blob(this->conv8_2_mbox_priorbox_datas_);

	destroy_blob(this->conv9_2_mbox_loc_datas_);
	destroy_blob(this->conv9_2_mbox_loc_perm_datas_);
	destroy_blob(this->conv9_2_mbox_loc_flat_datas_);
	destroy_blob(this->conv9_2_mbox_conf_datas_);
	destroy_blob(this->conv9_2_mbox_conf_perm_datas_);
	destroy_blob(this->conv9_2_mbox_conf_flat_datas_);
	destroy_blob(this->conv9_2_mbox_priorbox_datas_);

	destroy_blob(this->mbox_loc_datas_);
	destroy_blob(this->mbox_conf_datas_);
	destroy_blob(this->mbox_priorbox_datas_);

	destroy_blob(this->mbox_conf_reshape_datas_);
	destroy_blob(this->mbox_conf_softmax_datas_);
	destroy_blob(this->mbox_conf_flatten_datas_);

	destroy_blob(this->detection_out_datas_);
}

void Lcnn::setMean()
{
	vector<float> values{ 104,117,123 };
	std::vector<cv::Mat> channels;
	for (int i = 0; i < target_channel; ++i) {
		cv::Mat channel(target_width, target_height, CV_32FC1, cv::Scalar(values[i]));
		channels.push_back(channel);
	}
	cv::merge(channels, mean_);
}

vector<vector<float> > Lcnn::detect(Mat &image)
{
	cv::Mat sample_resize;
	cv::resize(image, sample_resize, cv::Size(300, 300));

	cv::Mat sample_float;
	sample_resize.convertTo(sample_float, CV_32FC3);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	Blob output = Forward(sample_normalized);

	const float* result = output.cpu_data();
	const int num_det = output.height_;
	vector<vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;
}

Blob Lcnn::Forward(Mat &inImg)
{
	//input
	input.forward_cpu(inImg, input_datas_);

	//conv1
	conv1_1.forward_cpu(input_datas_, conv1_1_datas_);
	relu1_1.forward_cpu(conv1_1_datas_);
	conv1_2.forward_cpu(conv1_1_datas_, conv1_2_datas_);
	relu1_2.forward_cpu(conv1_2_datas_);
	pool1.forward_cpu(conv1_2_datas_, pool1_datas_);

	//conv2
	conv2_1.forward_cpu(pool1_datas_, conv2_1_datas_);
	relu2_1.forward_cpu(conv2_1_datas_);
	conv2_2.forward_cpu(conv2_1_datas_, conv2_2_datas_);
	relu2_2.forward_cpu(conv2_2_datas_);
	pool2.forward_cpu(conv2_2_datas_, pool2_datas_);

	//conv3
	conv3_1.forward_cpu(pool2_datas_, conv3_1_datas_);
	relu3_1.forward_cpu(conv3_1_datas_);
	conv3_2.forward_cpu(conv3_1_datas_, conv3_2_datas_);
	relu3_2.forward_cpu(conv3_2_datas_);
	conv3_3.forward_cpu(conv3_2_datas_, conv3_3_datas_);
	relu3_3.forward_cpu(conv3_3_datas_);
	pool3.forward_cpu(conv3_3_datas_, pool3_datas_);

	//conv4
	conv4_1.forward_cpu(pool3_datas_, conv4_1_datas_);
	relu4_1.forward_cpu(conv4_1_datas_);
	conv4_2.forward_cpu(conv4_1_datas_, conv4_2_datas_);
	relu4_2.forward_cpu(conv4_2_datas_);
	conv4_3.forward_cpu(conv4_2_datas_, conv4_3_datas_);
	relu4_3.forward_cpu(conv4_3_datas_);
	pool4.forward_cpu(conv4_3_datas_, pool4_datas_);

	//conv5
	conv5_1.forward_cpu(pool4_datas_, conv5_1_datas_);
	relu5_1.forward_cpu(conv5_1_datas_);
	conv5_2.forward_cpu(conv5_1_datas_, conv5_2_datas_);
	relu5_2.forward_cpu(conv5_2_datas_);
	conv5_3.forward_cpu(conv5_2_datas_, conv5_3_datas_);
	relu5_3.forward_cpu(conv5_3_datas_);
	pool5.forward_cpu(conv5_3_datas_, pool5_datas_);

	//fc6
	fc6.forward_cpu(pool5_datas_, fc6_datas_);
	relu6.forward_cpu(fc6_datas_);

	//fc7
	fc7.forward_cpu(fc6_datas_, fc7_datas_);
	relu7.forward_cpu(fc7_datas_);

	//conv6
	conv6_1.forward_cpu(fc7_datas_, conv6_1_datas_);
	relu6_1.forward_cpu(conv6_1_datas_);
	conv6_2.forward_cpu(conv6_1_datas_, conv6_2_datas_);
	relu6_2.forward_cpu(conv6_2_datas_);

	//conv7
	conv7_1.forward_cpu(conv6_2_datas_, conv7_1_datas_);
	relu7_1.forward_cpu(conv7_1_datas_);
	conv7_2.forward_cpu(conv7_1_datas_, conv7_2_datas_);
	relu7_2.forward_cpu(conv7_2_datas_);

	//conv8
	conv8_1.forward_cpu(conv7_2_datas_, conv8_1_datas_);
	relu8_1.forward_cpu(conv8_1_datas_);
	conv8_2.forward_cpu(conv8_1_datas_, conv8_2_datas_);
	relu8_2.forward_cpu(conv8_2_datas_);

	//conv9
	conv9_1.forward_cpu(conv8_2_datas_, conv9_1_datas_);
	relu9_1.forward_cpu(conv9_1_datas_);
	conv9_2.forward_cpu(conv9_1_datas_, conv9_2_datas_);
	relu9_2.forward_cpu(conv9_2_datas_);

	//conv4_3_norm
	conv4_3_norm.forward_cpu(conv4_3_datas_, conv4_3_norm_datas_);
	//conv4_3_norm_mbox_loc
	conv4_3_norm_mbox_loc.forward_cpu(conv4_3_norm_datas_, conv4_3_norm_mbox_loc_datas_);
	//conv4_3_norm_mbox_loc_perm
	conv4_3_norm_mbox_loc_perm.forward_cpu(conv4_3_norm_mbox_loc_datas_, conv4_3_norm_mbox_loc_perm_datas_);
	//conv4_3_norm_mbox_loc_flat
	conv4_3_norm_mbox_loc_flat.forward_cpu(conv4_3_norm_mbox_loc_perm_datas_, conv4_3_norm_mbox_loc_flat_datas_);
	//conv4_3_norm_mbox_conf
	conv4_3_norm_mbox_conf.forward_cpu(conv4_3_norm_datas_, conv4_3_norm_mbox_conf_datas_);
	//conv4_3_norm_mbox_conf_perm
	conv4_3_norm_mbox_conf_perm.forward_cpu(conv4_3_norm_mbox_conf_datas_, conv4_3_norm_mbox_conf_perm_datas_);
	//conv4_3_norm_mbox_conf_flat
	conv4_3_norm_mbox_conf_flat.forward_cpu(conv4_3_norm_mbox_conf_perm_datas_, conv4_3_norm_mbox_conf_flat_datas_);
	//conv4_3_norm_mbox_priorbox
	conv4_3_norm_mbox_priorbox.forward_cpu(conv4_3_priorbox_inputs, conv4_3_norm_mbox_priorbox_datas_);

	//fc7_mbox_loc
	fc7_mbox_loc.forward_cpu(fc7_datas_, fc7_mbox_loc_datas_);
	//fc7_mbox_loc_perm
	fc7_mbox_loc_perm.forward_cpu(fc7_mbox_loc_datas_, fc7_mbox_loc_perm_datas_);
	//fc7_mbox_loc_flat
	fc7_mbox_loc_flat.forward_cpu(fc7_mbox_loc_perm_datas_, fc7_mbox_loc_flat_datas_);
	//fc7_mbox_conf
	fc7_mbox_conf.forward_cpu(fc7_datas_, fc7_mbox_conf_datas_);
	//fc7_mbox_conf_perm
	fc7_mbox_conf_perm.forward_cpu(fc7_mbox_conf_datas_, fc7_mbox_conf_perm_datas_);
	//fc7_mbox_conf_flat
	fc7_mbox_conf_flat.forward_cpu(fc7_mbox_conf_perm_datas_, fc7_mbox_conf_flat_datas_);
	//fc7_mbox_priorbox
	fc7_mbox_priorbox.forward_cpu(fc7_priorbox_inputs, fc7_mbox_priorbox_datas_);

	//conv6_2_mbox_loc
	conv6_2_mbox_loc.forward_cpu(conv6_2_datas_, conv6_2_mbox_loc_datas_);
	//conv6_2_mbox_loc_perm
	conv6_2_mbox_loc_perm.forward_cpu(conv6_2_mbox_loc_datas_, conv6_2_mbox_loc_perm_datas_);
	//conv6_2_mbox_loc_flat
	conv6_2_mbox_loc_flat.forward_cpu(conv6_2_mbox_loc_perm_datas_, conv6_2_mbox_loc_flat_datas_);
	//conv6_2_mbox_conf
	conv6_2_mbox_conf.forward_cpu(conv6_2_datas_, conv6_2_mbox_conf_datas_);
	//conv6_2_mbox_conf_perm
	conv6_2_mbox_conf_perm.forward_cpu(conv6_2_mbox_conf_datas_, conv6_2_mbox_conf_perm_datas_);
	//conv6_2_mbox_conf_flat
	conv6_2_mbox_conf_flat.forward_cpu(conv6_2_mbox_conf_perm_datas_, conv6_2_mbox_conf_flat_datas_);
	//conv6_2_mbox_priorbox
	conv6_2_mbox_priorbox.forward_cpu(conv6_2_priorbox_inputs, conv6_2_mbox_priorbox_datas_);

	//conv7_2_mbox_loc
	conv7_2_mbox_loc.forward_cpu(conv7_2_datas_, conv7_2_mbox_loc_datas_);
	//conv7_2_mbox_loc_perm
	conv7_2_mbox_loc_perm.forward_cpu(conv7_2_mbox_loc_datas_, conv7_2_mbox_loc_perm_datas_);
	//conv7_2_mbox_loc_flat
	conv7_2_mbox_loc_flat.forward_cpu(conv7_2_mbox_loc_perm_datas_, conv7_2_mbox_loc_flat_datas_);
	//conv7_2_mbox_conf
	conv7_2_mbox_conf.forward_cpu(conv7_2_datas_, conv7_2_mbox_conf_datas_);
	//conv7_2_mbox_conf_perm
	conv7_2_mbox_conf_perm.forward_cpu(conv7_2_mbox_conf_datas_, conv7_2_mbox_conf_perm_datas_);
	//conv7_2_mbox_conf_flat
	conv7_2_mbox_conf_flat.forward_cpu(conv7_2_mbox_conf_perm_datas_, conv7_2_mbox_conf_flat_datas_);
	//conv7_2_mbox_priorbox
	conv7_2_mbox_priorbox.forward_cpu(conv7_2_priorbox_inputs, conv7_2_mbox_priorbox_datas_);

	//conv8_2_mbox_loc
	conv8_2_mbox_loc.forward_cpu(conv8_2_datas_, conv8_2_mbox_loc_datas_);
	//conv8_2_mbox_loc_perm
	conv8_2_mbox_loc_perm.forward_cpu(conv8_2_mbox_loc_datas_, conv8_2_mbox_loc_perm_datas_);
	//conv8_2_mbox_loc_flat
	conv8_2_mbox_loc_flat.forward_cpu(conv8_2_mbox_loc_perm_datas_, conv8_2_mbox_loc_flat_datas_);
	//conv8_2_mbox_conf
	conv8_2_mbox_conf.forward_cpu(conv8_2_datas_, conv8_2_mbox_conf_datas_);
	//conv8_2_mbox_conf_perm
	conv8_2_mbox_conf_perm.forward_cpu(conv8_2_mbox_conf_datas_, conv8_2_mbox_conf_perm_datas_);
	//conv8_2_mbox_conf_flat
	conv8_2_mbox_conf_flat.forward_cpu(conv8_2_mbox_conf_perm_datas_, conv8_2_mbox_conf_flat_datas_);
	//conv8_2_mbox_priorbox
	conv8_2_mbox_priorbox.forward_cpu(conv8_2_priorbox_inputs, conv8_2_mbox_priorbox_datas_);

	//conv9_2_mbox_loc
	conv9_2_mbox_loc.forward_cpu(conv9_2_datas_, conv9_2_mbox_loc_datas_);
	//conv9_2_mbox_loc_perm
	conv9_2_mbox_loc_perm.forward_cpu(conv9_2_mbox_loc_datas_, conv9_2_mbox_loc_perm_datas_);
	//conv9_2_mbox_loc_flat
	conv9_2_mbox_loc_flat.forward_cpu(conv9_2_mbox_loc_perm_datas_, conv9_2_mbox_loc_flat_datas_);
	//conv9_2_mbox_conf
	conv9_2_mbox_conf.forward_cpu(conv9_2_datas_, conv9_2_mbox_conf_datas_);
	//conv9_2_mbox_conf_perm
	conv9_2_mbox_conf_perm.forward_cpu(conv9_2_mbox_conf_datas_, conv9_2_mbox_conf_perm_datas_);
	//conv9_2_mbox_conf_flat
	conv9_2_mbox_conf_flat.forward_cpu(conv9_2_mbox_conf_perm_datas_, conv9_2_mbox_conf_flat_datas_);
	//conv9_2_mbox_priorbox
	conv9_2_mbox_priorbox.forward_cpu(conv9_2_priorbox_inputs, conv9_2_mbox_priorbox_datas_);

	mbox_loc.forward_cpu(mbox_loc_inputs, mbox_loc_datas_);
	mbox_conf.forward_cpu(mbox_conf_inputs, mbox_conf_datas_);
	mbox_priorbox.forward_cpu(mbox_priorbox_inputs, mbox_priorbox_datas_);

	mbox_conf_reshape.forward_cpu(mbox_conf_datas_, mbox_conf_reshape_datas_);
	mbox_conf_softmax.forward_cpu(mbox_conf_reshape_datas_, mbox_conf_softmax_datas_);
	mbox_conf_flatten.forward_cpu(mbox_conf_softmax_datas_, mbox_conf_flatten_datas_);

	detection_out.forward_cpu(detection_out_inputs, detection_out_datas_);

	return detection_out_datas_[0];
}