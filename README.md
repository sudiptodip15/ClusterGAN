# ClusterGAN

Code for reproducing key results in the paper [ClusterGAN : Latent Space Clustering in Generative Adversarial Networks](https://arxiv.org/abs/1809.03627) by Sudipto Mukherjee, Himanshu Asnani, Eugene Lin and Sreeram Kannan. If you use the code, please cite our paper.

## Dependencies 

The code has been tested with the following versions of packages.
- Python 2.7.12
- Tensorflow 1.4.0
- Numpy 1.14.2

## Datasets

The datasets used in the paper can be downloaded from the Google Drive link(https://drive.google.com/open?id=1XnGkSamF5DiwnpHFG0OexmoqAwe27ucR).
Unzip the folder so that the path is : ./ClusterGAN/data/<dataset_name>

##Training

You can either train your own models on the datasets or use pre-trained models. Even though we have used a fixed seed using tf.random.seed(0), there will still be randomness introduced by CUDA. So, to reproduce the results, train 5 models and compare the Validation purity in the logs directory. Each model can be trained as follows :

```bash
$ python Image_Cluster.py --data mnist --K 10 --dz 30 --beta_n 10 --beta_c 10 --train True 

This will save the model along with timestamp in checkpoint-dir/<dataset_name>. Also, the Validation set performance will be written to logs/Res_<dataset_name>_<model_name>.txt. Then run the best model (with highest Validation Purity) on the Test set. 

```bash
$ python Image_ClusterGAN.py --data mnist --K 10 --dz 30 --beta_n 10 --beta_c 10 --timestamp <best_timestamp>

Training the models for other datasets has a similar format.
Fashion-10 : 
```bash
$ python Image_ClusterGAN.py --data fashion --K 10 --dz 40 --beta_n 0 --beta_c 10 --train True 

Fashion-5 : 
```bash
$ python Image_ClusterGAN.py --data fashion --K 5 --dz 40 --beta_n 0 --beta_c 10 --train True 

Single Cell 10x genomics : 
```bash
$ python Gene_ClusterGAN.py --data 10x_73k --K 8 --dz 30 --beta_n 10 --beta_c 10 --train True 

Pendigits : 
```bash
$ python pen_ClusterGAN.py --data pendigit --K 10 --dz 5 --beta_n 10 --beta_c 10 --train True 

Provide the timestamp of best saved model to obtain the Test set clustering performance on all the datasets (similar to MNIST above).

##Pre-trained models

Additionally, you can also download the pre-trained models from the Google drive link(https://drive.google.com/open?id=1l9Lwq0amAaA3qHzNCiw7BrivSAFoP0em). Unzip the file in ./ClusterGAN. It should lead to the folder ./ClusterGAN/pre_trained_models

Run the following code : 

$ python Image_ClusterGAN.py --data mnist --K 10 --dz 30 --beta_n 10 --beta_c 10 

Similarly for the other datasets.

##Clustering Performance

Table shows the mean +- standard deviation of 10 runs of ClusterGAN (with the reported hyperparameter settings in the paper) for various datasets.

|    Dataset    |           ACC       |         NMI         |           ARI         |
|:-------------:|:-------------------:|:-------------------:|:---------------------:|
|      MNIST    | 0.9097 +- 0.0398 | 0.8544 +- 0.0361 | 0.8290 +- 0.0621   |
|   Fashion-10  | 0.6119 +- 0.0230 | 0.6157 +- 0.0112 | 0.4617 +- 0.0226   |
|   Fashion-5   | 0.7218 +- 0.0089 | 0.6163 +- 0.0243 | 0.5035 +- 0.0228   |
|    10x_73k    | 0.8172 +- 0.0262 | 0.7272 +- 0.0322 | 0.6786 +- 0.0369   |
|   Pendigits   | 0.7638 +- 0.0120 | 0.7343 +- 0.0120 | 0.6336 +- 0.0177   |


##Feedback

Please feel free to provide any feedback about the code to sudipm@uw.edu.


