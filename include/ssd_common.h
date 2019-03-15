#ifndef SSD_COMMON_H
#define SSD_COMMON_H

#include "input_layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "relu_layer.h"
#include "normalize_layer.h"
#include "permute_layer.h"
#include "flatten_layer.h"
#include "prior_box_layer.h"
#include "concat_layer.h"
#include "reshape_layer.h"
#include "softmax_layer.h"
#include "detection_output_layer.h"

void readToInputLayer(InputLayer &input, FILE* fp);
void readToConvLayer(ConvolutionLayer &conv, FILE* fp);
void readToPoolingLayer(PoolingLayer &pool, FILE *fp);
void readToNormalizeLayer(NormalizeLayer &norm, FILE*fp);
void readToPriorBoxLayer(PriorBoxLayer &priorbox, FILE*fp);
void readToConcatLayer(ConcatLayer &concat, int axis);

void destroy_blob(vector<Blob>& des_datas_);
void destroy_blob_single(Blob& des_datas_);

#endif