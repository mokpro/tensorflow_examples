# Solving MNIST Using TensorFlow

I started using the [beginner](https://www.tensorflow.org/tutorials/mnist/beginners/) tutorial.


These steps more like what I did rather than what should be done! Also some notes along with it.


## Follow [MNIST For ML Beginners](https://www.tensorflow.org/tutorials/mnist/beginners/)

### Steps:

1. Get the MNIST [data](http://yann.lecun.com/exdb/mnist/).
  - It is a dataset of handwritten digits, and has been processed to fit in 28x28 pixel box
  - It combines census bureau employees and high schoolers data.
  - Training vs testing writers are disjoint
  - The image files are actually vector files which tells you how the image looks, without showing it.
  - The the [import](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py#L37) is likely to take couple of minutes to execute as it downloads data. Logs:

    ```
    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    ```

2. Running:
  - We setup a softmax layer in neural network (?)
  - Define a loss (or cost) function
  - Choose an optimizer with aim to minimize the loss function
  - Run tf code for 1000 iterations and print accuracy

Accuracy output with [beginner_mnist.py](./beginner_mnist.py):
```
(u'GradientDescent', 0.91540003)
(u'Adagrad', 0.92089999)
(u'Adadelta', 0.92470002)
(u'ProximalAdagrad', 0.91939998)
(u'ProximalGradientDescent', 0.92290002)
```
