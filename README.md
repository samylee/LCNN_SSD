# LCNN_SSD
Forward C++ Code of the SSD based on CAFFE, CPU support only.
# Description
1. The model is modified based on [SSD300_07+12](https://github.com/weiliu89/caffe/tree/ssd) by weiliu89. The efficiency of the model is the same as that of the original model.
2. C++ code ONLY!
3. WHITOUT backward!
4. Based on [CAFFE](https://github.com/BVLC/caffe).
# Algorithm efficiency
| Image Size | Speed | mAP(voc2007-test) | CPU | Compiler |
|:------:|:------:|:------:|:------:|:------:|
| Any resolution image  | 800ms |0.772| i7-4790-4core @3.6GHz | VS2015 |
# Test steps
## step1
Download model and put it to "model/". [BaiDu CLoud](https://pan.baidu.com/s/1Jj1EwPc3D_9rh7l63Ex2Zw), `PassWord: tvn9`
## step2
Compile with `opencv` and `openblas`, then test.
# Example result
![image](https://github.com/samylee/LCNN_SSD/blob/master/images/000001.png)
![image](https://github.com/samylee/LCNN_SSD/blob/master/images/000004.png)
# Reference
https://blog.csdn.net/samylee
