Repository for Deep Learning Project for Image Recognition
==============

----
Bare Minimum
----
Can we reduce CNN training effort by a novel proprocessing? : Traditionally only affine / distortion transformation for data augmentation. 

1. Visualization - Data Exploration / Understanding the success of CNN for plankton/galaxy dataset:
    - traditional dataset has been pretty well explored in terms of visualization
    - (inspired by successes of CNN for plankton/galaxy kaggle competition) Visualize hidden layer for model trained by plankton data / galaxy data - using deconvnet
    - If (2) works Visualizing hidden layer for the model trained with original data versus the model trained with reduced data. Giving reasoning why dimensional reduction works (or not)

2. Explored dimension reduction by the method presented in 'Learning Transformation for Clustering and Classification'
    - This paper has explored dimensional reduction using subspace transformation and use nearst neighbor/[] to perform image classification. (face clustering, motion segmentation)
    - Question: Can this method be used as a preprocessing step for CNN? (transform original data to reduced data)
    - Effort:


--- Extension ---
3. Boosting CNN
    - Boost CNN (AdaBoost) and visualize hidden layer
    - Challenge: The number of parameters can be very large for multiple CNNs. Extremely nontrivial if we can't fit the parameters in GPU memory.
    - 

----
Research Focus
---- 
1. Dimensional reduction / subspace projection as a preprocessing for neural network
    Concern: what if the pre-processed data performs so poorly compared using original data

2. Visualize hidden layers for boosted CNNs
    Concern: there might not be enough differences between each CNN. Will the result be revealing? the subsequent CNNs are essentially trained 

3. Maybe studying autoencoder for Plankton or galaxy dataset? Not very novel but a baby project


----
Discarded ideas
----
1. Exploring the hidden layer for rotational invariant training data
    - This has been done by Lenc K. and Vadaldi A.
    - Question: Can we apply a method to an existing dataset? (galaxy and plankton are rotationally invariant data)
2. 





Reference Research Papers

[0]He Kaiming et at.
- Used parametric rectified activation units in neural net + robust initialization method
26% improvement over 2014 winner (4.94 versus 6.66%) Supassed human-level performance (5.1%)!!!


[13] A. G. Howard. Some improvements on deep convolutional neural network based image classification. arXiv:1312.5402, 2013.
- This paper addes image transformation to training data (scale, view, color, crop) - 20% better than previous year winner



[12] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov. Improving neural networks by pre- venting co-adaptation of feature detectors. arXiv:1207.0580, 2012.
- Used 'dropout' to prevent complex co-adaptation (meaning ?). Set new records on speech and object recognition.
- Dataset - TIMIT: acoustic phonetic continuous speech corpus
- Reuters corpus volume
- CIFAR-10 : CNN
- ImageNet : CNN
- This paper has great summary of CNNs



[20] A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In ICML, 2013.
- mentioned that Gloret et al. 2011 found that DNNs with rectifier nonlinearities perform better than sigmoidal model for image recognition and classification task
- compared the hidden representations of rectifier and sigmoidal networks : offers insight as to why rectifier nonlinearities perform well. 

(old paper - ImageNet)
[16] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet clas- sification with deep convolutional neural networks. In NIPS, 2012.
- very large scale system: 15.3 % error rate
- 


[22] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. arXiv:1409.0575, 2014.
- TO READ


=============
Possible Research Agendas
=============
1. Explores dimensionality reduction - Learning transformations: Qiu Q. Mar 2014
    - Find newer paper?
2. Other pre-processing:
    - Find other paper as well?

3. Neural Network Tweaking
    - Activation function
      - learned the rectifier unit 
  
4. Understanding neural network

5. Rotation: 
 - what property of neural network can make it invariant to rotation
 - Galaxy or plankton data:
 - How can they be generalized for actual images (CIFAR-10/100, Caltech-101)


Random ideas:
  - auto-encoder: 
  
Tools:
Code for deconvolutioning neural network? 

Data:
Galaxy Data, Kaggle Plankton Competition http://www.kaggle.com/c/datasciencebowl
MNIST
CIFAR-10, CIFAR-100
Caltech 101
Caltech 256
SVHN : Street View House Number


http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

