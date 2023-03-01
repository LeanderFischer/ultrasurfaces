"""
Calculate both probabilities and gradients for a given set of systematic datasets using the GradientNetwork classifier.
"""

from typing import List
import pandas as pd
import numpy as np
import keras

from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from .keras_models import GradientNetwork
from .calculate_grads import make_delta_p_from_grad_names, make_gradient_names

# Run the network given a list of toy_mc.Generator objects

def run_grad_net(
    nominal_dataset,
    sys_datasets,
    variables: List[str],
    param_names: List[str],
    poly_order: int = 2,
    hidden_layer_sizes: tuple = (100, 100),
    activation: str = "relu",
    # verbose can be "auto", 0, 1, or 2
    verbose: int = 0,
    epochs: int = 10,
    shuffle: bool = True,
    learning_rate: float = 1e-5,
):
    """
    Run the GradientNetwork on a set of nominal and systematic datasets.

    Parameters
    ----------
    nominal_dataset : Generator
        The nominal dataset that we want to calculate the probabilities for.
    sys_datasets : list of Generator
        A list of systematic datasets that we want to use to calculate the probabilities.
    variables : list of str
        A list of the variables that we want to use as input features in the classifier.
    poly_order : int, optional
        The order of the polynomial to use in the classifier (default is 2).
    param_names : list of str
        A list of the parameters that we want to calculate gradients for.
    hidden_layer_sizes : tuple, optional
        The number of neurons in each hidden layer of the classifier (default is (100, 100)).
    activation : str, optional
        The activation function to use in the classifier (default is "relu").
    verbose : int, optional
        The verbosity level of the classifier (default is 0).
    epochs : int, optional
        The number of epochs to train the classifier for (default is 10).
    shuffle : bool, optional
        Whether to shuffle the data before each epoch (default is True).
    learning_rate : float, optional
        The learning rate to use in the classifier (default is 1e-5).
    """
    sys_names = [dataset.name for dataset in sys_datasets]
    for name in sys_names:
        assert name is not None, "must define names for sys sets"
    assert nominal_dataset.name is not None, "must define name of nominal set"
    assert len(sys_names) == len(set(sys_names)), "sys set names must be unique"
    assert nominal_dataset.name not in sys_names, "nominal set in sys sets"

    # Get the DataFrame with the events from the nominal dataset.
    df_nominal = nominal_dataset.events
    df_nominal["set"] = nominal_dataset.name
    
    # Make combined DataFrame out of all the events in the systematic and nominal sets.
    sys_dataframes = [df_nominal]
    for dataset in sys_datasets:
        df = dataset.events
        df["set"] = dataset.name
        sys_dataframes.append(df)
    df_comb = pd.concat(sys_dataframes, ignore_index=True)
    df_comb["set"] = pd.Categorical(df_comb["set"], sys_names + [nominal_dataset.name])
    # A transformer to make all of the input data normal
    data_encoder = make_column_transformer(
        (
            preprocessing.PowerTransformer(method="box-cox", standardize=True),
            variables,
        ),
        remainder="drop",
    )
    # label transformer
    label_encoder = preprocessing.LabelEncoder().fit(df_comb["set"])
    X = data_encoder.fit_transform(df_comb)
    y_index = label_encoder.transform(df_comb["set"])
    # convert to one-hot encoding
    y = np.zeros((len(y_index), len(label_encoder.classes_)))
    y[np.arange(len(y_index)), y_index] = 1

    # Configure the network
    # First, get names of gradients
    gradient_names = make_gradient_names(include_systematics=param_names, poly_features=poly_order)
    # Then, get the delta_p matrix
    delta_p = make_delta_p_from_grad_names(gradient_names, sys_datasets, nominal_dataset)
    n_grads = delta_p.shape[0]
    assert n_grads == len(gradient_names)
    # Finally, make the network
    grad_net = GradientNetwork(
        hidden_layer_sizes=hidden_layer_sizes,
        delta_p=delta_p,
        activations=[activation] * len(hidden_layer_sizes),
        dropout_rates=[0.0] * len(hidden_layer_sizes),
    )
    # Compile and fit the network using Adam with given learning rate
    grad_net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
    )
    grad_net.fit(X, y, epochs=epochs, batch_size=1000, verbose=verbose, shuffle=shuffle)

    # create one column in the data frame for the probability of each set
    prob_cols = [f"prob_{set_label}" for set_label in label_encoder.classes_]

    df_nominal[prob_cols] = 0

    X_nominal = data_encoder.transform(df_nominal)
    probs = grad_net.predict(X_nominal)
    df_nominal[prob_cols] = probs

    # to get gradients, evaluate the sub-network that calculates the gradients
    gradients_evaluated = grad_net.grad_net.predict(X_nominal)
    # store the gradients in the data frame
    df_nominal[gradient_names] = gradients_evaluated
    
    return df_nominal
