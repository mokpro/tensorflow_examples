# Solving MNIST Using TensorFlow

I started using the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/).


These steps more like what I did rather than what should be done! Also some notes along with it.


## Follow [MNIST For ML Beginners](https://www.tensorflow.org/tutorials/mnist/beginners/)

### Steps:

1. Get the MNIST [data](http://yann.lecun.com/exdb/mnist/).
  - It is a dataset of handwritten digits, and has been processed to fit in 28x28 pixel box
  - It combines census bureau employees and high schoolers data.
  - Training vs testing writers are disjoint
  - The image files are actually vector files which tells you how the image looks, without showing it.
  - The the [import](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py#L37) is likely to take couple of minutes to execute as it downloads data.

2. Running:
  - We setup a softmax layer in neural network (?)
  - Define a loss (or cost) function
  - Choose an optimizer with aim to minimize the loss function
    ```bash
      $ python beginner_mnist.py Adagrad
    ```
  - Run tf code for 1000 iterations and print accuracy

Accuracy output with [beginner_mnist.py](./beginner_mnist.py) for 5 optimizers:

```bash
  $ python beginner_mnist.py --all
  Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
  Extracting MNIST_data/train-images-idx3-ubyte.gz
  Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
  Extracting MNIST_data/train-labels-idx1-ubyte.gz
  Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
  Extracting MNIST_data/t10k-images-idx3-ubyte.gz
  Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
  Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

  Optimizer: GradientDescent, Accuracy: 0.8983
  Optimizer: Adagrad, Accuracy: 0.9163
  Optimizer: Adadelta, Accuracy: 0.918
  Optimizer: ProximalAdagrad, Accuracy: 0.9209
  Optimizer: ProximalGradientDescent, Accuracy: 0.9216
```


## Follow [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

### Steps:

  1. Do same steps as above and create [expert_mnist.py](./expert_mnist.py)
  2. Refactor to use `InteractiveSession()` instead of `Session()`
  3. Optimization Steps - TBD
  4. We get higher accuracy. (How?)

## Notes
- Used [autopep8](https://pypi.python.org/pypi/autopep8) to format python
