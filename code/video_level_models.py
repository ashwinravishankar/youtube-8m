
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 8,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_integer(
    "coarse_num_mixtures", 8,
    "The number of coarse classification mixtures (excluding the dummy 'expert') used for HMoeModel.")

flags.DEFINE_integer(
    "label_num_mixtures", 8, 
    "The number of label classification mixtures (excluding the dummy 'expert') used for HMoeModel.")

class FCNeuralNetworkModel(models.BaseModel):
    """It is simple 2 layer neural network with L2 regularization and relu activation. Output layer is softmax"""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""
        print("Model used: Simple 2 layer neural network")

        h1 = slim.fully_connected(model_input, int(model_input.get_shape().as_list()[-1] / 2), activation_fn=tf.nn.relu,
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))

        output = slim.fully_connected(h1, vocab_size, activation_fn=tf.nn.softmax,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}


class BranchedNNModel(models.BaseModel):
    """FFNN model with individual di red of audio and video"""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        print("Model used: Branched 'V' neural network")
        # separate the input vector of size 1152 into video and audio part 
        # video,audio = tf.split(model_input,[1024,128],axis=1)
        audio = model_input[:, -128:]
        video = model_input[:, :-128]

        # dimensionality reduction

        # FC layer to reduce the dimensions of video
        vdNN = slim.fully_connected(video, 512, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # a FC layer to reduce the dimensions of video; 
        adNN = slim.fully_connected(audio, 64, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # concatenate;
        mix = tf.concat([vdNN,adNN],axis=1)

        # FC layer;
        h1 = slim.fully_connected(mix, int(mix.get_shape().as_list()[-1]/2), activation_fn=tf.nn.relu,
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))

        # final softmax layer for classification; 
        output = slim.fully_connected(h1, vocab_size, activation_fn=tf.nn.softmax,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))

        print(output)
        return {"predictions": output}


class CNNModel(models.BaseModel):
    """CNN model as extension of VGG net """

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        print("Model used: CNN model")
        print(model_input.get_shape())
        # video,audio = tf.split(model_input,[1024,128],axis=1)[0]
        audio = model_input[:, -128:]
        video = model_input[:, :-128]
        print("video shape: ",video.shape)
        print("audio shape: ",audio.shape)

        # dimensionality reduction
        
        vdNN = slim.fully_connected(video, 32, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        adNN = slim.fully_connected(audio, 32, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # calculate outer product of video and audio
        vd = tf.expand_dims(vdNN,-1)

        ad = tf.expand_dims(adNN,-1)

        ad = tf.transpose(ad,perm=[0,2,1])

        # calculate outer product; matrix multiplication
        mix = tf.matmul(vd,ad)

        mix = tf.expand_dims(mix,-1)

        # first convolutional layer; 
        conv1 = tf.layers.conv2d(inputs=mix, filters=8, kernel_size=[3,3])

        # average pooling;
        avgpool = tf.layers.average_pooling2d(inputs=conv1, pool_size=2,strides=2)

        # second convolutional layer; 
        conv2 = tf.layers.conv2d(inputs=avgpool, filters=4, kernel_size=[3,3])

        # flatten the output;
        flat = tf.contrib.layers.flatten(inputs=conv2)

        # final softmax layer for classification;
        output = slim.fully_connected(flat, vocab_size, activation_fn=tf.nn.softmax,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))

        print(output)
        return {"predictions": output}


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1])) 
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures])) 

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class HMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   coarse_num_mixtures=None,
                   label_num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   phase=True,
                   **unused_params):
    coarse_num_mixtures = coarse_num_mixtures or FLAGS.coarse_num_mixtures
    label_num_mixtures = label_num_mixtures or FLAGS.label_num_mixtures

    ### Layer 1
    coarse_gate_activations = slim.fully_connected(
        model_input,
        25 * (coarse_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_gates")
    coarse_expert_activations = slim.fully_connected(
        model_input,
        25 * coarse_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="coarse_experts")

    coarse_gating_distribution = tf.nn.softmax(tf.reshape(
        coarse_gate_activations,
        [-1, coarse_num_mixtures + 1]))  
    coarse_expert_distribution = tf.nn.sigmoid(tf.reshape(
        coarse_expert_activations,
        [-1, coarse_num_mixtures]))  


    coarse_probabilities_by_class_and_batch = tf.reduce_sum(
        coarse_gating_distribution[:, :coarse_num_mixtures] * coarse_expert_distribution, 1)
    coarse_probabilities = tf.reshape(coarse_probabilities_by_class_and_batch,
                                     [-1, 25])

    concat = tf.concat([model_input, coarse_probabilities], -1, name = 'middle_concat')

    ### Layer 2
    label_gate_activations = slim.fully_connected(
        concat,
        vocab_size * (label_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_gates")
    label_expert_activations = slim.fully_connected(
        concat,
        vocab_size * label_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="label_experts")

    label_gating_distribution = tf.nn.softmax(tf.reshape(
        label_gate_activations,
        [-1, label_num_mixtures + 1]))  
    label_expert_distribution = tf.nn.sigmoid(tf.reshape(
        label_expert_activations,
        [-1, label_num_mixtures]))  

    label_probabilities_by_class_and_batch = tf.reduce_sum(
        label_gating_distribution[:, :label_num_mixtures] * label_expert_distribution, 1)
    label_probabilities = tf.reshape(label_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    return {"predictions": label_probabilities}

class ResModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, is_training=True, l2_penalty=1e-8, **unused_params):

        if is_training == True:
            drop_prob = 0.5
        else:
            drop_prob = 1

        input_layer = slim.fully_connected(model_input, 10000, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        drop_layer_1 = slim.dropout(input_layer, drop_prob)
        hidden_layer = slim.fully_connected(drop_layer_1, 10000, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        drop_layer_2 = slim.dropout(hidden_layer, drop_prob)
        skip_layer = tf.add(input_layer, drop_layer_2)
        output = slim.fully_connected(skip_layer, vocab_size, activation_fn=tf.nn.softmax, weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}

