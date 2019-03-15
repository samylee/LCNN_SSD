#include "input_layer.h"

void InputLayer::layer_set_up(vector<Blob>& top_data)
{
	//保留第一张图像，其余删除
	top_data.resize(1);
	top_data[0].reshape(1, this->input_params_.channel_, this->input_params_.height_, this->input_params_.width_);
}

void InputLayer::forward_cpu(Mat &inputImg, vector<Blob>& top_data)
{
	if (this->input_params_.channel_ == 3)
	{
		float* imgData = (float*)malloc(sizeof(float) * this->input_params_.height_ * this->input_params_.width_ * 3);

		int index = 0;
		for (int c = 0; c < this->input_params_.channel_; c++)
		{
			for (int nr = 0; nr < this->input_params_.height_; nr++)
			{
				for (int nc = 0; nc < this->input_params_.width_; nc++)
				{
					imgData[index] = inputImg.at<Vec3f>(nr, nc)[c];
					index += 1;
				}
			}
		}

		top_data[0].set_data(imgData);
		free(imgData);
	}
	else
	{
		cout << "The image's channel should be 3, please check it!" << endl;
		system("pause");
		exit(0);
	}
}