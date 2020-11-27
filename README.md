# tf_unsupervised_count
_Unofficial_ Tensorflow implementation of **Representation Learning by Learning to Count** 



### TL;DR

## Requirements

* Python 3+ [Tested on 3.6.10]
* Tensorflow 1.X [Tested on 1.14]
* TensorPack [Tested on 0.10.1]
* Nvidia-DALI [Tested on 0.20.0]
* Nvidia-DALI-Plugin [Tested on 0.20.0]

We use cuda 10.0.130 and cudnn v7.6.5

Our TensorFlow model and loss function are simple. However, to train it *efficiently*, we needed to use TensorPack and Nvidia-DALI libraries. If linking these libraries is challenging, feel free to remove them from the code and use Tensorflow only (e.g., tf.data.dataset). The requirements.txt lists all our install packages and their versions.

## ImageNet Pretrained Models

## Usage example

| ImageNet Performance          | conv1 | conv2 | conv3 | conv4 | conv5 |
|-------------------------------|-------|-------|-------|-------|-------|
| Mehdi et at. [1] \(Table. 2\) | 18.0  | 30.6  | 34.3  | 32.5  | 25.7  |
| Ours                          |       |       |       |       |       |
    
### TODO LIST


Contributor list
----------------
1. [Ahmed Taha](http://ahmed-taha.com/)
2. Alex Hanson

**It would be great if someone re-implement this in pytorch. Let me know and I will add a link to your Pytorch implementation here**


### MISC Notes
* Our implementation is inspired by [CLVR's implementation](https://github.com/clvrai/Representation-Learning-by-Learning-to-Count). However, the CLVR's implementation has a serious bug and performance issues that need to be fixed. These issues are discussed [here](https://github.com/ahmdtaha/tf_unsupervised_count/blob/main/docs/clvr_bug.md).
* Our implementation diverges from the paper [1] technical details. We explain why this discrepancy is required [here](https://github.com/ahmdtaha/tf_unsupervised_count/blob/main/docs/paper_discrepancy.md).

## Release History
* 1.0.0
    * First commit on 25 Nov 2020


## References
[1] Representation Learning by Learning to Count
