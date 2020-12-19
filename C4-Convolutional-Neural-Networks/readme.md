# Course 4: Convolutional Neural Networks

- [Course 4: Convolutional Neural Networks](#course-4-convolutional-neural-networks)
  - [Week 1: Foundations of Convolutional Neural Networks](#week-1-foundations-of-convolutional-neural-networks)
    - [Learning Objectives](#learning-objectives)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
      - [Computer Vision](#computer-vision)
      - [Edge Detection Example](#edge-detection-example)
      - [More Edge Detection](#more-edge-detection)
      - [Padding](#padding)
      - [Strided Convolutions](#strided-convolutions)
      - [Convolutions Over Volume](#convolutions-over-volume)
      - [One Layer of a Convolutional Network](#one-layer-of-a-convolutional-network)
      - [Simple Convolutional Network](#simple-convolutional-network)
      - [Pooling Layers](#pooling-layers)
      - [CNN Example](#cnn-example)
      - [Why Convolutions](#why-convolutions)

## Week 1: Foundations of Convolutional Neural Networks

### Learning Objectives

- Explain the convolution operation
- Apply two different types of pooling operations
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build and train a ConvNet in TensorFlow for a classification problem

### Convolutional Neural Networks

#### Computer Vision

*Deep learning computer vision can now*:

- help self-driving cars figure out where the other cars and pedestrians around so as to avoid them.
- make face recognition work much better than ever before.
- unlock a phone or unlock a door using just your face.

*Deep learning for computer vision is exciting* because:

- First, rapid advances in computer vision are enabling brand new applications to view, though they just were impossible a few years ago.
- Second, even if you don't end up building computer vision systems per se, I found that because the computer vision research community has been so creative and so inventive in coming up with new neural network architectures and algorithms, is actually inspire that creates a lot cross-fertilization into other areas as well.

For computer vision applications, you don't want to be stuck using only tiny little images. You want to use large images. To do that, you need to better implement the **convolution operation**, which is one of the fundamental building blocks of **convolutional neural networks**.

#### Edge Detection Example

- The convolution operation is one of the fundamental building blocks of a convolutional neural network.
- Early layers of the neural network might detect edges and then some later layers might detect parts of objects and then even later layers may detect parts of complete objects like people's faces.
- Given a picture for a computer to figure out what are the objects in the picture, the first thing you might do is maybe detect edges in the image.

The *convolution operation* gives you a convenient way to specify how to find these **vertical edges** in an image.

A `3 by 3` filter or `3 by 3` matrix may look like below, and this is called a vertical edge detector or a vertical edge detection filter. In this matrix, pixels are relatively bright on the left part and relatively dark on the right part.

```text
1, 0, -1
1, 0, -1
1, 0, -1
```

Convolving it with the vertical edge detection filter results in detecting the vertical edge down the middle of the image. 

![edge-detection](img/edge-detect-v.png)

#### More Edge Detection

In the horizontal filter matrix below, pixels are relatively bright on the top part and relatively dark on the bottom part.

```text
 1,  1,  1
 0,  0,  0
-1, -1, -1
```

Different filters allow you to find vertical and horizontal edges. The following filter is called a **Sobel filter** the advantage of which is it puts a little bit more weight to the central row, the central pixel, and this makes it maybe a little bit more robust. [More about Sobel filter](https://fiveko.com/tutorials/image-processing/sobel-filter/).

```text
1, 0, -1
2, 0, -2
1, 0, -1
```

Here is another filter called **Scharr filter**:

```text
 3, 0, -3
10, 0, -10
 3, 0, -3
```

More about [**Scharr filter**](https://plantcv.readthedocs.io/en/v3.0.5/scharr_filter/).


```text
w1, w2, w3
w4, w5, w6
w7, w8, w9
```

By just letting all of these numbers be parameters and learning them automatically from data, we find that neural networks can actually learn low level features, can learn features such as edges, even more robustly than computer vision researchers are generally able to code up these things by hand.

#### Padding

In order to fix the following two problems, padding is usually applied in the convolutional operation.

- Every time you apply a convolutional operator the image shrinks.
- A lot of information from the edges of the image is thrown away.

*Notations*:

- image size: `n x n`
- convolution size: `f x f`
- padding size: `p`

*Output size after convolution*:

- without padding: `(n-f+1) x (n-f+1)`
- with padding: `(n+2p-f+1) x (n+2p-f+1)`

*Convention*:

- Valid convolutions: no padding
- Same convolutions: output size is the same as the input size
- `f` is usually odd

#### Strided Convolutions

*Notation*:

- stride `s`

*Output size after convolution*: `floor((n+2p-f)/s+1) x floor((n+2p-f)/s+1)`

*Conventions*:

- The filter must lie entirely within the image or the image plus the padding region.
- In the deep learning literature by convention, a convolutional operation (maybe better *called cross-correlation*) is what we usually do not bother with a flipping operation, which is included before the product and summing step in a typical math textbook or a signal processing textbook.
  - In the latter case, the filter is flipped vertically and horizontally.

#### Convolutions Over Volume

For a RGB image, the filter itself has three layers corresponding to the red, green, and blue channels.

`height x width x channel`

`n x n x nc` * `f x f x nc` --> `(n-f+1) x (n-f+1) x nc'`

#### One Layer of a Convolutional Network

*Notations*:

| size | notation |
| :---- | :---- |
| filter size | ![f(l)](img/layer_filter_size.svg) |
| padding size | ![p(l)](img/layer_padding_size.svg) |
| stride size | ![s(l)](img/layer_stride_size.svg) |
| number of filters | ![nc(l)](img/layer_num_filters.svg) |
| filter shape | ![filter_shape](img/layer_filter_shape.svg) |
| input shape | ![input_shape](img/layer_input_shape.svg) |
| output shape | ![output_shape](img/layer_output_shape.svg) |
| output height | ![nh(l)](img/layer_output_height.svg) |
| output width | ![nw(l)](img/layer_output_width.svg) |
| activations `a[l]` | ![activations](img/layer_output_shape.svg) |
| activations `A[l]` | ![activations](img/layer_activations.svg) |
| weights | ![weights](img/layer_weights.svg) |
| bias | ![bias](img/layer_bias.svg) |

#### Simple Convolutional Network

Types of layer in a convolutional network:

- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)

#### Pooling Layers

- One interesting property of max pooling is that it has a set of hyperparameters but it has no parameters to learn. There's actually nothing for gradient descent to learn.
- Formulas that we had developed previously for figuring out the output size for conv layer also work for max pooling.
- The max pooling is used much more often than the average pooling.
- When you do max pooling, usually, you do not use any padding.

#### CNN Example
 
- Because the pooling layer has no weights, has no parameters, only a few hyper parameters, I'm going to use a convention that `CONV1` and `POOL1` shared together.
- As you go deeper usually the *height* and *width* will decrease, whereas the number of *channels* will increase.
- max pooling layers don't have any parameters
- The conv layers tend to have relatively few parameters and a lot of the parameters tend to be in the fully collected layers of the neural network.
- The activation size tends to maybe go down *gradually* as you go deeper in the neural network. If it drops too quickly, that's usually not great for performance as well.

![nn-example](img/nn-example.png)

*Layer shapes of the network*:

| layer | activation shape | activation size | # parameters |
| :----: | :----: | :----: | :----: |
| Input | (32,32,3) | 3072 | 0 |
| CONV1 (f=5,s=1) | (28,28,8) | 6272 | 608 `=(5*5*3+1)*8` |
| *POOL1* | (14,14,8) | 1,568 | 0 |
| CONV2 (f=5,s=1) | (10,10,16) | 1600 | 3216 `=(5*5*8+1)*16` |
| *POOL2* | (5,5,16) | 400 | 0 |
| FC3 | (120,1) | 120 | 48120 `=400*120+120` |
| FC4 | (84,1) | 84 | 10164 `=120*84+84` |
| softmax | (10,1) | 10 | 850 `=84*10+10` |

#### Why Convolutions

There are two main advantages of convolutional layers over just using fully connected layers.

- Parameter sharing: A feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image.
- Sparsity of connections: In each layer, each output value depends only on a small number of inputs.

Through these two mechanisms, a neural network has a lot fewer parameters which allows it to be trained with smaller training cells and is less prone to be overfitting.

- Convolutional structure helps the neural network encode the fact that an image shifted a few pixels should result in pretty similar features and should probably be assigned the same output label.
- And the fact that you are applying the same filter in all the positions of the image, both in the early layers and in the late layers that helps a neural network automatically learn to be more robust or to better capture the desirable property of translation invariance. 
