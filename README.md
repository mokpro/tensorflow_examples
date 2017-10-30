# tensorflow_examples
Trying out tensorflow and solving example problems.

# Installation Guidelines
## For MacOS (El Captain)

The following installation steps are cherry-picked from [TensorFlow Installation Guide](https://www.tensorflow.org/get_started/):

1. Check if you are using system python

  ```bash
    $ which python
  ```
2. If output is `/usr/local/bin/python` then skip to step 4.
3. Install a python at user level using [Homebrew](http://brew.sh/):

  ```bash
    $ brew install python
    $ brew linkapps python
    $ which python
    /usr/local/bin/python
  ```
4. Install or update [pip](https://pip.pypa.io/en/stable/) using [easy_install](http://setuptools.readthedocs.io/en/latest/easy_install.html):

  ```bash
    $ easy_install pip
    $ pip install --upgrade pip
  ```
5. Install [TensorFlow](https://www.tensorflow.org):

  ```bash
    $ pip install tensorflow
    $ pip install tensorflow-gpu # Optional
  ```
6. Test the installation by loading tensorflow in a python console:

  ```bash
    $ python
    >>> import tensorflow as tf
  ```

# Resources to explore

1. [TensorFlow tutorials](https://www.tensorflow.org/tutorials/)
2. [Learning TensorFlow](http://learningtensorflow.com/)
3. [TF Learn](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn) (previously SkFlow)
