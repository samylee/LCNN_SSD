#include "relu_layer.h"

ReluLayer::ReluLayer()
{
}

ReluLayer::~ReluLayer()
{
}

void ReluLayer::forward_cpu(vector<Blob>& data_blob)
{
	float* bottom_data = data_blob[0].cpu_data();
	const int count = data_blob[0].counts_;
	for (int i = 0; i < count; ++i) {
		bottom_data[i] = std::max(bottom_data[i], float(0));
	}
}