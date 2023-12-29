from itertools import product
import os

import numpy as np


def prepare_train(
        data_dir_models,
        batch_factors,
        do_weights,
        model_names,
        losses,
        learning_rates,
        moderates,
        Number_of_KFolds,
        Number_of_KFolds_to_run,
):

    # Calculate number of folds to be run
    NFolds = min(Number_of_KFolds, Number_of_KFolds_to_run)

    # List of pre-trained DBN models
    dbn_models = [os.path.join(data_dir_models, file) for file in model_names]

    # Create grid for intepretation of the results
    params = {
        "batch_factor": batch_factors,
        "do_weight": do_weights,
        "dbn_model": dbn_models,
        "loss": losses,
        "learning_rate": learning_rates,
        "moderate": moderates,
    }
    grid = list(product(*params.values()))
    grid_with_names = [tuple(zip(params.keys(), values)) for values in grid]

    #%%
    # Initialize mae_list
    params_with_folds = {
        "batch_factor": batch_factors,
        "fold": range(0, NFolds),
        "do_weight": do_weights,
        "dbn_model": dbn_models,
        "loss": losses,
        "learning_rate": learning_rates,
        "moderate": moderates,
    }
    grid_with_folds = list(product(*params_with_folds.values()))
    grid_with_folds_with_names = [
        tuple(zip(params_with_folds.keys(), values)) for values in grid_with_folds
    ]
    
    shape = tuple(len(value) for value in params_with_folds.values())

    return (
        NFolds,
        grid_with_names,
        dbn_models,
        grid,
        grid_with_folds_with_names,
        shape,
    )

def prepare_train_noise(
        data_dir_models,
        batch_factors,
        do_weights,
        model_names,
        losses,
        learning_rates,
        moderates,
        noises,
        Number_of_KFolds,
        Number_of_KFolds_to_run,
):

    # Calculate number of folds to be run
    NFolds = min(Number_of_KFolds, Number_of_KFolds_to_run)

    # Create grid for intepretation of the results
    params = {
        "batch_factor": batch_factors,
        "do_weight": do_weights,
        "dbn_model": dbn_models,
        "loss": losses,
        "learning_rate": learning_rates,
        "moderate": moderates,
        "noise": noises,
    }
    grid = list(product(*params.values()))
    grid_with_names = [tuple(zip(params.keys(), values)) for values in grid]

    # List of pre-trained DBN models
    dbn_models = [os.path.join(data_dir_models, file) for file in model_names]

    #%%
    # Initialize mae_list
    params_with_folds = {
        "batch_factor": batch_factors,
        "fold": range(0, NFolds),
        "do_weight": do_weights,
        "dbn_model": dbn_models,
        "loss": losses,
        "learning_rate": learning_rates,
        "moderate": moderates,
        "noise": noises,
    }
    grid_with_folds = list(product(*params_with_folds.values()))
    grid_with_folds_with_names = [
        tuple(zip(params_with_folds.keys(), values)) for values in grid_with_folds
    ]
    
    shape = tuple(len(value) for value in params_with_folds.values())

    return (
        NFolds,
        grid_with_names,
        dbn_models,
        grid,
        grid_with_folds_with_names,
        shape,
    )