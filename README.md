Quick Draw Recognition
--

## Introduction
This project is about somthing

## TEFPA
T = Classify object

E = data set 

F = 

P = categorical_crossentropy

A =  CNN, ...

## Installation 

Install 

```pip install -r requirements.txt```

Run app server

```python app.py```


## Data

Source: https://quickdraw.withgoogle.com/data/

![image](images/sample1.png)

Class: ```Watermelon```

![image](images/sample2.png)

Class: ```Effel Tower```


## Model
I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

![model](images/models.png)

## Summary Table
|      | |
| ---------- |-------------------|
| **Author**       | Le Hung Son|
| **Title**        | Quick Draw Recognition |
| **Topics**       | Ứng dụng trong computer vision, làm game, sử dụng thuật toán chính là CNN|
| **Descriptions** | Input là danh sách các numpy array được lưu trong file ```*.npy```, tên file chính là tên object tương ứng. Khi train xong sẽ trả ra output là file trọng số ```doodle_cnn_model.hdf5```. Ta sẽ sử dụng trọng số này đã train để predict.|
| **Links**        | https://github.com/sonlh07/final-vtc-quick-draw |
| **Framework**    | Tensorflow|
| **Pretrained Models**  | |
| **Datasets**     |Mô hình được train với bộ dữ liệu https://github.com/googlecreativelab/quickdraw-dataset|
| **Level of difficulty**|Sử dụng nhanh và dễ, có thể train lại với tập dữ liệu khác tốc độ tùy thuộc vào phần cứng và hình ảnh input|


## Training & validation

Train the model:

```python train.py```

All trained weights will saved to:

```saved/*.hdf5```

![training](images/train.png)


## Demo

1. Draw an object on the right board
2. Click the Predict button
3. Results will be showed
(You can repeat these steps with the clear button)

![Demo](images/demo.png)

## References
1. https://github.com/googlecreativelab/quickdraw-dataset#get-the-data
2. https://qz.com/994486/the-way-you-draw-circles-says-a-lot-about-you/
3. https://github.com/lutzroeder/netron
4. https://quickdraw.withgoogle.com/data/watermelon
5. https://github.com/gary30404/convolutional-neural-network-from-scratch-python/
