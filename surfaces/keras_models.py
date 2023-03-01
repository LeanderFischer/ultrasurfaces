"""
Keras models that produce gradients and make predictions for the re-weighting of MC events
"""

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers
import numpy as np
from collections import OrderedDict


def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)


class Swish(layers.Layer):
    """
    Swish activation layer with trainable beta parameter.

    See also: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    """

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = beta
        self.trainable = trainable
        # Signal that the layer is safe for mask propagation.
        # Probably not needed here, but just to be sure:
        # https://keras.io/guides/understanding_masking_and_padding/
        self.supports_masking = True

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name="beta_factor")
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = super(Swish, self).get_config()
        config.update(
            {
                "beta": self.get_weights()[0] if self.trainable else self.beta,
                "trainable": self.trainable,
            }
        )
        return config


class GradientNetwork(keras.Model):
    def __init__(
        self,
        hidden_layer_sizes,
        activations,
        dropout_rates,
        delta_p,
        weight_decay=1e-5,
        k_regularization=1e-5,
        gradient_regularization=0.0,
        with_class_bias=False,
    ):

        self.delta_p = tf.constant(np.array(delta_p).astype(np.float32))

        # there will be one gradient for each polynomial feature
        n_gradients = self.delta_p.shape[0]

        super(GradientNetwork, self).__init__()

        # Build the network that generates gradients for each event
        self.grad_net = keras.Sequential(name="gradient_generator")
        for i, n in enumerate(hidden_layer_sizes):
            self.grad_net.add(
                keras.layers.Dense(
                    n,
                    # For swish, we will add the activation as a separate layer
                    # to get a trainable beta parameter.
                    activation=None if activations[i] == "swish" else activations[i],
                    kernel_regularizer=keras.regularizers.l2(weight_decay),
                    name=f"hidden_layer_{i}",
                )
            )
            # add swish activation separately
            if activations[i] == "swish":
                self.grad_net.add(Swish(beta=1.0, trainable=True))

            if dropout_rates[i] > 0.0:
                self.grad_net.add(
                    keras.layers.Dropout(dropout_rates[i], name=f"dropout_layer_{i}")
                )

        self.grad_net.add(
            keras.layers.Dense(
                n_gradients,
                name="raw_gradients",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                activity_regularizer=keras.regularizers.l2(gradient_regularization),
            )
        )

        # we give the network the freedom to re-scale and rotate the raw gradients
        # so that it can deal with different parameters working on different scales
        # (such as DOM efficiency, hole ice p0, hole ice p1)
        self.grad_net.add(
            keras.layers.Dense(
                n_gradients,
                name="gradient_rescale_and_bias",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(k_regularization),
                kernel_initializer=keras.initializers.Identity(gain=1.0),
            )
        )

        self.class_bias = tf.Variable(
            initial_value=tf.zeros(self.delta_p.shape[1]), trainable=with_class_bias
        )
        self.prob_floor = tf.Variable(initial_value=0.01, trainable=False)

        # we store all the arguments that were used to create this model so that
        # we can save and recover it with get_config and from_config
        self._serializable_state = OrderedDict(
            hidden_layer_sizes=hidden_layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            delta_p=delta_p,
            weight_decay=weight_decay,
            k_regularization=k_regularization,
            gradient_regularization=gradient_regularization,
            with_class_bias=with_class_bias,
        )

    def get_config(self):
        return self._serializable_state

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        grads = self.grad_net(x)
        # We have the freedom to add an arbitrary class bias that we later neglect
        # when we re-weight events. Since the total normalization is free in the
        # real analysis, an offset in normalization should have no negative impact
        # on the fit.
        x = tf.matmul(grads, self.delta_p) + self.class_bias

        x = tf.nn.softmax(x)

        return x

# define test functions for the GradientNetwork to be used by pytest
def test_gradient_network():
    from numpy.testing import assert_allclose
    # test that the GradientNetwork can be created and called
    # without crashing
    delta_p = np.random.normal(size=(10, 5))
    model = GradientNetwork(
        hidden_layer_sizes=[10, 10],
        activations=["swish", "swish"],
        dropout_rates=[0.0, 0.0],
        delta_p=delta_p,
        weight_decay=1e-5,
        k_regularization=1e-5,
        gradient_regularization=0.0,
        with_class_bias=False,
    )

    x = np.random.normal(size=(100, 10))
    y = model(x)

    assert y.shape == (100, 5)

    # test that the GradientNetwork can be saved and restored
    # without crashing
    config = model.get_config()
    model = GradientNetwork.from_config(config)
    y = model(x)

    assert y.shape == (100, 5)

    # Make a simple test dataset consisting of three classes of normally
    # distributed events with different means.
    # The mean of each class should be a linear function of a parameter "mu"
    # that is different for each class. The GradientNetwork should find
    # the first and second-order gradients with respect to "mu". Therefore,
    # the delta_p matrix has to be constructed accordingly (see below).

    mu = np.array([-0.1, 0.0, 0.1])
    sigma = 0.5
    n_events = 10000
    n_features = 3
    n_classes = len(mu)
    # Two gradients, one for first order and one for second order
    n_grads = 2

    # sample events for each class from a normal distribution
    # with the mean given by the mu parameter
    x = []
    # y should be one-hot encoded labels for the three classes
    y = []
    # set seed for numpy to make test reproducible
    np.random.seed(0)
    
    for i, mu_class in enumerate(mu):
        x.append(np.random.normal(size=(n_events, n_features), scale=sigma, loc=mu_class))
        y_class = np.zeros((n_events, n_classes))
        y_class[:, i] = 1.0
        y.append(y_class)
    
    x = np.array(x)
    y = np.array(y)

    # The shape of delta_p is (n_gradients, n_classes).
    # The output of the network is the gradients matrix-multiplied with delta_p.
    # The delta_p matrix contains the difference in mu and mu-squared for each class.

    delta_p = np.zeros((n_grads, n_classes))
    delta_p[0, :] = mu
    delta_p[1, :] = mu ** 2

    # construct the model
    model = GradientNetwork(
        hidden_layer_sizes=[10, 10],
        activations=["swish", "swish"],
        dropout_rates=[0.0, 0.0],
        delta_p=delta_p,
        weight_decay=1e-5,
        k_regularization=1e-5,
        gradient_regularization=0.0,
        with_class_bias=False,
    )

    
    # compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    # evaluate the score of the model before training
    score_before = model.evaluate(x, y, batch_size=1000)
    # train the model
    model.fit(x, y, epochs=10, batch_size=1000)
    # evaluate the score of the model after training
    score_after = model.evaluate(x, y, batch_size=1000)
    # the score should be better after training
    assert score_after[1] > score_before[1]



    


    