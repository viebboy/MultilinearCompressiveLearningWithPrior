# Multilinear Compressive Learning with Prior Knowledge
arXiv:

What is Multilinear Compressive Learning?
------------------------------------------

[Compressive Learning (CL)](https://arxiv.org/abs/1610.09615) is a learning paradigm in which learning models are built on top of compressed measurements obtained from Compressive Sensing devices. Since the measurements (often small number) are obtained by linear interpolation of the analog signal, there are limitations regarding the amount and the kind of information that can be captured from the analog signal via Compressive Sensing. Thus, to sucessfully build learning models on top of compressed measurements are more difficult compared to the case where learning models are built on top of high-fidelity data. 

While traditional CL models combine vector-based Compressive Sensing (regardless the type of data) and machine learning, [Multilinear Compressive Learning](https://arxiv.org/abs/1905.07481) combines machine learning and Multilinear Compressive Sensing to harness the tensor structures of certain types of data. MCL results in better computational complexity and learning performances. 

Learning with Prior Knowledge
-------------------------------

Because only a small fraction of the signal is captured via compressed measurements, the straightforward questions to ask when performing CL or MCL are:

* What kind of information we should capture in the compressed measurements that are beneficial for the learning task?
* What kind of features we should synthesize from the compressed measurements for the learning task?

The prior knowledge about such information can guide the learning models towards better solutions. In this work, we propose a technique to **find** and **incorporate** prior knowledge into MCL models. We also propose a semi-supervised adaptation that allows the technique to take advantages of unlabeled data. Through extensive experiments, we demonstrate that the proposed technique can really lead the MCL models to better solutions. Although we only limit our investigation to MCL models, the proposed technique can work with any end-to-end CL models. 

Code
-----

The code is written using Keras with Tensorflow backend. The datasets with train/val/test split can be downloaded from [here](https://bit.ly/2m7CaXW). After cloning our repository and downloading the data, the data should be put in the directory named "data" at the same level as the "code" directory, i.e.,::

	MultilinearCompressiveLearningWithPrior/code

	MultilinearCompressiveLearningWithPrior/data


To run all the experiment configurations, simply run::

	bash train.sh

Any question related to this work, please contact: thanh.tran@tuni.fi or viebboy@gmail.com


