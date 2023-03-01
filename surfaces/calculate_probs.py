"""
Run computation of event-wise probabilities and store the results in a Pandas DataFrame.
"""
import os
import re
from typing import List
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from .sample_weighted_kneighbors import SampleWeightedKNeighborsClassifier
# import mlp classifier from scikit-learn
from sklearn.neural_network import MLPClassifier


def calculate_probs_sklearn(
    nominal_dataset: pd.DataFrame,
    sys_datasets: List[pd.DataFrame],
    variables: List[str],
    use_weights: bool = False,
    classifiertype: str = "knn",
    # options for the MLP classifier
    hidden_layer_sizes: tuple = (100, 100),
    verbose_training: bool = False,
    max_iter: int = 10,
    alpha: float = 1e-6,
    activation: str = "relu",
    # options for the knn classifier
    neighbors_per_class: int = 100,
    jobs: int = 1,
    splits: int = 10,
    tilt_bias_correction: bool = True,
) -> pd.DataFrame:
    """
    Calculate the probability of each set by using a (modified) scikit-learn classifier.
    The options for classifiers are either the sample-weighted KNN or the regular MLP 
    classifier from scikit-learn. The sample-weighted KNN classifier is used by default.

    Parameters
    ----------
    nominal_dataset : pandas DataFrame
        The nominal dataset that we want to calculate the probabilities for.
    sys_datasets : list of pandas DataFrame
        A list of systematic datasets that we want to use to calculate the probabilities.
    variables : list of str
        A list of the variables that we want to use as input features in the KNN classifier.
    use_weights : bool, optional
        Whether to to use event weights when making neighbors calculation (default is False).
        If False, raw MC events are used to calculate probabilities.
    classifiertype : str, optional
        The type of classifier to use. Options are "knn" or "mlp" (default is "knn").
    hidden_layer_sizes : tuple, optional
        The number of neurons in each hidden layer of the MLP classifier (default is (100, 100)).
    neighbors_per_class : int, optional
        The number of neighbors per class to use in the KNN classifier (default is 100).
    jobs : int, optional
        The number of parallel jobs to run when fitting the KNN classifier (default is 1).
    splits : int, optional
        The number of chunks to split the data frame into (default is 10).
    tilt_bias_correction : bool, optional
        Whether to perform tilt and bias correction in the KNN classifier (default is True).

    Returns
    -------
    pandas DataFrame
        A DataFrame with the probability of each set for each event in the nominal dataset.
    """

    # make sure that the classifiertype is either knn or mlp
    assert classifiertype in ["knn", "mlp"], "classifiertype must be either knn or mlp"
    # mlp cannot use weights
    assert not (classifiertype == "mlp" and use_weights), "mlp cannot use weights"
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
    y = label_encoder.transform(df_comb["set"])
    if use_weights:
        weights = df_comb["weights"].to_numpy() / np.mean(df_comb["weights"])
    else:
        weights = np.ones(len(df_comb))

    # Fit the classifier
    if classifiertype == "mlp":
        # MLP classifier
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=False,
            random_state=42,
            verbose=verbose_training,
            alpha=alpha,
            activation=activation,
        )
    else:
        # KNN classifier
        classifier = SampleWeightedKNeighborsClassifier(
            n_neighbors=neighbors_per_class * (len(sys_datasets) + 1),
            n_jobs=jobs,
            weights="uniform",
        )
    if use_weights:
        classifier.fit(X, y, weights)
    else:
        classifier.fit(X, y)

    # create one column in the data frame for the probability of each set
    prob_cols = [f"prob_{set_label}" for set_label in label_encoder.classes_]

    df_nominal[prob_cols] = 0

    # Evaluate the classifier
    # Only for the KNN classifier, we use chunking to avoid memory allocation errors
    # as well as tilt-bias correction.
    if classifiertype == "knn":
        print("Starting KNN evaluation...")
        # We split the data frame into chunks to avoid memory allocation errors
        split_df = np.array_split(df_nominal, splits)
        # Also use tqdm to make a nice progress bar, since this is going to take a while...
        for df in tqdm(split_df):
            X_query = data_encoder.transform(df)
            # Evaluate the classifier
            # only for the KNN classifier, we can use the tilt and bias correction
            probs = classifier.predict_proba(
                X_query,
                correct_bias=tilt_bias_correction,
            )
            # Write to data frame
            df[prob_cols] = probs
        # putting all the splits back together should produce a DataFrame where every row
        # has probabilities for all categories in the appropriate columns
        df_nominal = pd.concat(split_df, ignore_index=True)
    else:
        # For the MLP classifier, we can just use the predict_proba method
        print("Starting MLP evaluation...")
        X_query = data_encoder.transform(df_nominal)
        probs = classifier.predict_proba(X_query)
        df_nominal[prob_cols] = probs
    
    return df_nominal

