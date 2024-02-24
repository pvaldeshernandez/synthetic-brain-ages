import time
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_class_weight, compute_sample_weight
from keras import layers
from keras import optimizers
from keras.utils import Sequence
from keras.models import clone_model, Model
from keras.callbacks import Callback, EarlyStopping
import statsmodels.formula.api as smf
import pingouin as pg


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_epoch_begin(self, batch, logs=None):
        if self.times:
            print(f"Epoch time: {self.times[-1]} seconds")
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs=None):
        if self.times:
            print(f"Epoch time: {self.times[-1]} seconds")


class StopOnOverfitting(Callback):
    def __init__(self, patience=0):
        super(StopOnOverfitting, self).__init__()
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_mae") > logs.get("mae"):
            if self.wait >= self.patience:
                print("\nStopping training as val_mae is higher than mae")
                self.stopped_epoch = epoch
                self.model.stop_training = True
            else:
                self.wait += 1
        else:
            self.wait = 0


def custom_transform(encoder, x):
    # Create a binary array with the same shape as the output of encoder.transform
    result = np.zeros((len(x), len(encoder.classes_)))
    # Find the indices of the values in x that are in encoder.classes_
    valid_indices = np.isin(x, encoder.classes_)
    # Transform only the valid values
    if np.any(valid_indices):
        result[valid_indices] = encoder.transform(x[valid_indices])

    return result


class CustomDataSequenceTwoInputsAndAge(Sequence):
    def __init__(
        self,
        gen,
        dataframe,
        x_col,
        y_col,
        flow_args,
        modalities,
        scanners,
        use_sample_weights=False,
    ):
        # Create a DataFrameIterator instance
        self.dataframe = dataframe
        self.dataflow1 = gen.flow_from_dataframe(
            dataframe=dataframe, x_col=x_col[0], y_col=y_col, **flow_args
        )

        # Extract the categorical data from the DataFrame
        self.x2 = dataframe[x_col[1]].values
        self.x3 = dataframe[x_col[2]].values
        self.age = dataframe[x_col[3]].values

        # Create a LabelBinarizer instance to encode the categorical data
        if modalities is None:
            modalities = array(["T2w-SR", "MPRAGE", "T1w-SR", "T2wFLAIR-SR", "MPRAGE-SR"])

        if scanners is None:
            scanners = array(
                ["Aera", "Verio", "Avanto", "Prisma", "Titan3T", "Sola"], dtype=object
            )
        self.encoder_modalities = LabelBinarizer()
        self.encoder_modalities.fit(modalities)
        self.encoder_scanners = LabelBinarizer()
        self.encoder_scanners.fit(scanners)

        # Store the use_sample_weights argument as an instance variable
        self.use_sample_weights = use_sample_weights

        if use_sample_weights:
            weights_dict = {}
            # age
            weights_dict["age"] = compute_sample_weight("balanced", dataframe["age"])
            # sex
            sexes = np.unique(dataframe["Sex"])
            sex_weights = compute_class_weight("balanced", classes=sexes, y=dataframe["Sex"])
            weights_dict["sex"] = np.array(
                [dict(zip(sexes, sex_weights))[i] for i in dataframe["Sex"]]
            )
            # ids
            ids = np.unique(dataframe["ID"])
            id_weights = compute_class_weight("balanced", classes=ids, y=dataframe["ID"])
            weights_dict["id"] = np.array([dict(zip(ids, id_weights))[i] for i in dataframe["ID"]])
            # modality
            modality_weights = compute_class_weight(
                "balanced", classes=modalities, y=dataframe["modality"]
            )
            weights_dict["modality"] = np.array(
                [dict(zip(modalities, modality_weights))[i] for i in dataframe["modality"]]
            )
            # scanner
            scanner_weights = compute_class_weight(
                "balanced", classes=scanners, y=dataframe["scanner"]
            )
            weights_dict["scanner"] = np.array(
                [dict(zip(scanners, scanner_weights))[i] for i in dataframe["scanner"]]
            )
            # weights
            self.sample_weights = np.ones_like(weights_dict[use_sample_weights[0]])
            for weight in use_sample_weights:
                self.sample_weights *= weights_dict[weight]

    def __len__(self):
        return len(self.dataflow1)

    def __getitem__(self, idx):
        # Get the next batch of data from dataflow1
        x1, y = self.dataflow1[idx]

        # Get the corresponding batch of categorical data
        x2 = self.x2[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]
        x3 = self.x3[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]

        # Get the corresponding batch of age
        age = self.age[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]

        # Transform the categorical data using the LabelBinarizer instance
        # x2 = self.encoder.transform(x2)
        x2 = custom_transform(self.encoder_modalities, x2)
        x3 = custom_transform(self.encoder_scanners, x3)

        if self.use_sample_weights:
            # Get the sample weights for the current batch
            sample_weights = self.sample_weights[
                idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size
            ]
            sample_weights = np.array(sample_weights)
            # Return a batch of data and sample weights
            return [x1, x2, x3, age], y, sample_weights
        else:
            # Return a batch of data without sample weights
            return [x1, x2, x3, age], y

    def on_epoch_end(self):
        # Shuffle the data in dataflow1
        self.dataflow1.on_epoch_end()

        # Check if shuffling is enabled for dataflow1
        if self.dataflow1.shuffle:
            # Get the shuffled index of the data
            index_array = self.dataflow1.index_array

            # Shuffle the categorical data using the same index
            self.x2 = self.x2[index_array]
            self.x3 = self.x3[index_array]

            # Shuffle age using the same index
            self.age = self.age[index_array]

            if self.use_sample_weights:
                # Shuffle the sample weights using the same index
                self.sample_weights = self.sample_weights[index_array]

                # if self.use_sample_weights == 5 and self.model is not None:
                #     # Get model predictions on the entire dataset
                #     y_pred = self.model.predict([self.dataflow1, self.x2, self.x3, self.age])

                #     # Compute MSE
                #     mse = mean_squared_error(self.age, y_pred)

                #     # Adjust weights based on MSE
                #     if mse > 100:  # set a suitable threshold
                #         self.sample_weights *= 0.9
                #     elif mse < 1:  # set a suitable threshold
                #         self.sample_weights *= 1.1

                #     # Ensure weights are still valid probabilities
                #     self.sample_weights = np.clip(self.sample_weights, 0, 1)
                #     self.sample_weights /= np.sum(self.sample_weights)

    def set_model(self, model):
        # Set the model for this sequence
        self.model = model


class CustomDataGeneratorTwoInputsAndAge:
    def __init__(
        self,
        gen,
        dataframe,
        x_col,
        y_col,
        flow_args,
        modalities,
        scanners,
        use_sample_weights=False,
    ):
        # Create a DataFrameIterator instance
        self.dataflow1 = gen.flow_from_dataframe(
            dataframe=dataframe, x_col=x_col[0], y_col=y_col, **flow_args
        )

        # Extract the categorical data from the DataFrame
        self.x2 = dataframe[x_col[1]].values
        self.x3 = dataframe[x_col[2]].values
        self.age = dataframe[x_col[3]].values

        # Extract age
        self.age = dataframe[x_col[2]].values

        # Create a LabelBinarizer instance to encode the categorical data
        self.encoder_modalities = LabelBinarizer()
        self.encoder_modalities.fit(modalities)
        self.encoder_scanners = LabelBinarizer()
        self.encoder_scanners.fit(scanners)

        if use_sample_weights:
            # Calculate the sample weights based on the class distribution
            self.sample_weights = compute_sample_weight("balanced", dataframe[y_col])

    def __iter__(self):
        while True:
            # Get the next batch of data from dataflow1
            x1, y = next(self.dataflow1)
            x2 = self.x2
            x3 = self.x3
            age = self.age

            # Get the corresponding batch of categorical data
            idx = self.dataflow1.batch_index - 1
            x2 = self.x2[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]
            x3 = self.x3[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]
            age = self.age[idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size]

            # Transform the categorical data using the LabelBinarizer instance
            x2 = custom_transform(self.encoder_modalities, x2)
            x3 = custom_transform(self.encoder_scanners, x3)

            if self.use_sample_weights:
                # Get the sample weights for the current batch
                sample_weights = self.sample_weights[
                    idx * self.dataflow1.batch_size : (idx + 1) * self.dataflow1.batch_size
                ]
                sample_weights = np.array(sample_weights)
                # Return a batch of data and sample weights
                yield [x1, x2, x3, age], y, sample_weights
            else:
                # Return a batch of data without sample weights
                yield [x1, x2, x3, age], y

    def on_epoch_end(self):
        # Shuffle the data in dataflow1
        self.dataflow1.on_epoch_end()

        # Check if shuffling is enabled for dataflow1
        if self.dataflow1.shuffle:
            # Get the shuffled index of the data
            index_array = self.dataflow1.index_array

            # Shuffle the categorical data using the same index
            self.x2 = self.x2[index_array]
            self.x3 = self.x3[index_array]

            # Shuffle age using the same index
            self.age = self.age[index_array]

            if self.use_sample_weights:
                # Shuffle the sample weights using the same index
                self.sample_weights = self.sample_weights[index_array]


def fit_with_transfer_learning(
    model,
    LAYERS_TO_TRAIN,
    loss,
    learning_rate,
    train_sequence,
    step_size_train,
    linear_sequence,
    step_size_linear,
    valid_sequence,
    step_size_valid,
    linear_df,
    formulas,
    decay=3e-7,
    epochs=None,
    workers=16,
    correction_method="clear",
):

    # Deifne the default number of epochs for each training phase
    if epochs is None:
        epochs = [20, 20]

    # Define the early stopping criteria
    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    stop_on_overfitting = StopOnOverfitting(patience=0)
    callbacks = [early_stop, stop_on_overfitting]

    # Get the time
    callbacks = callbacks.copy()
    callbacks.append(TimeHistory())

    # Set all layers but the last 'layers_to_train' to trainable=False
    for layer in model.layers[:-LAYERS_TO_TRAIN]:
        layer.trainable = False

    # Create a dictionary to store the combined training history
    history = {}
    times = []

    # Compile with selected learning rate and loss
    model.compile(
        loss=loss,
        optimizer=optimizers.Adam(learning_rate=learning_rate, decay=decay),
        metrics=["mae"],
    )

    # Fit the model
    if valid_sequence is not None and step_size_valid is not None:
        history1 = model.fit(
            x=train_sequence,
            steps_per_epoch=step_size_train,
            validation_data=valid_sequence,
            validation_steps=step_size_valid,
            epochs=epochs[0],
            workers=workers,
            callbacks=callbacks,
        )

    else:
        history1 = model.fit(
            x=train_sequence,
            steps_per_epoch=step_size_train,
            epochs=epochs[0],
            workers=workers,
            callbacks=callbacks,
        )

    # Mark the time
    times.append(callbacks[-1].times)

    # Update the combined training history
    for key, value in history1.history.items():
        history[key] = value
    actual_epochs = [max([callback.stopped_epoch for callback in callbacks[:-1]])]

    # Set all layers to trainable=False
    for layer in model.layers[:]:
        layer.trainable = True

    # Compile with selected learning rate and loss
    model.compile(
        loss=loss,
        optimizer=optimizers.Adam(learning_rate=learning_rate, decay=decay),
        metrics=["mae"],
    )

    # Fit the model
    if valid_sequence is not None and step_size_valid is not None:
        history2 = model.fit(
            x=train_sequence,
            steps_per_epoch=step_size_train,
            validation_data=valid_sequence,
            validation_steps=step_size_valid,
            epochs=epochs[1],
            workers=workers,
            callbacks=callbacks,
        )
    else:
        history2 = model.fit(
            x=train_sequence,
            steps_per_epoch=step_size_train,
            epochs=epochs[1],
            workers=workers,
            callbacks=callbacks,
        )

    # Mark the time
    times.append(callbacks[-1].times)

    # Update the combined training history
    for key, value in history2.history.items():
        history[key].extend(value)
    actual_epochs.append(max([callback.stopped_epoch for callback in callbacks[:-1]]))

    # Predict all brain ages to calculate brain age per subject
    linear_brainage_slices = model.predict(linear_sequence, verbose=1, steps=step_size_linear)

    # Add the predictions as a new column in test_df
    linear_df = linear_df.copy()
    linear_df.loc[:, "brainage_slices"] = linear_brainage_slices  # [0].item()

    # Calculate agregated dataframe
    linear_grouped_agg = group_by_ID(linear_df)

    # Correct "regression dilution" bias
    corrected_models = []
    results = []
    for formula in formulas:
        corrected_model, result = add_linear_age_correction_to_model(
            model, linear_grouped_agg, formula[0], correction_method, formula[1]
        )
        corrected_models.append(corrected_model)
        results.append(result)

    return corrected_models, results, actual_epochs, history, times


def add_linear_age_correction_to_model(
    model, linear_grouped_agg, formula, correction_method, do_wls
):
    # Estimate linear regression parameters
    results, _ = estimate_correction(linear_grouped_agg, formula, do_wls)

    # Get model's inputs (image, modality, scanner, and age only)
    inputs = model.input[:4]

    # Add the custom layer to the model that takes both the model's input and 'age' and/or
    params = results.params.values
    corrected_brainage = keras.Lambda(
        lambda x: correct_brainage(x, params, correction_method, formula)
    )([model.output] + inputs[1:])

    corrected_model = Model(inputs=inputs, outputs=corrected_brainage)

    return corrected_model, results


def estimate_correction(linear_grouped_agg, formula, do_wls):
    if formula == "brainage ~ age":
        modified_formula = formula
    elif formula == "brainage ~ age ^ 2":
        modified_formula = "brainage ~ age + I(age**2)"
    elif formula == "brainage ~ age * modality":
        modified_formula = "brainage ~ age * modality - age - 1"
    elif formula == "brainage ~ age ^ 2 * modality":
        modified_formula = "brainage ~ age * modality + I(age**2) * modality - I(age**2) - age - 1"
    elif formula == "brainage ~ age * scanner":
        modified_formula = "brainage ~ age * scanner - age - 1"
    elif formula == "brainage ~ age ^ 2 * scanner":
        modified_formula = "brainage ~ age * scanner + I(age**2) * scanner - I(age**2) - age - 1"
    elif formula == "brainage ~ age * modality * scanner":
        modified_formula = (
            "brainage ~ age * modality * scanner - age * modality - age * scanner - 1"
        )
    elif formula == "brainage ~ age ^ 2 * modality * scanner":
        modified_formula = (
            "brainage ~ age * modality * scanner + I(age**2) * modality * scanner"
            " - I(age**2) * modality - I(age**2) * scanner - age * modality - age * scanner - 1"
        )

    if do_wls:
        # Initialize weights as None
        weights = None

        # Check if 'modality' is in the formula
        if "modality" in modified_formula:
            weights_modality = 1 / linear_grouped_agg["modality"].map(
                linear_grouped_agg["modality"].value_counts()
            )
            weights = weights_modality if weights is None else weights + weights_modality

        # Check if 'scanner' is in the formula
        if "scanner" in modified_formula:
            weights_scanner = 1 / linear_grouped_agg["scanner"].map(
                linear_grouped_agg["scanner"].value_counts()
            )
            weights = weights_scanner if weights is None else weights + weights_scanner

        # Normalize the weights to sum up to 1
        if weights is not None:
            weights /= weights.sum()
    else:
        weights = None

    # Fit the model
    if weights is None:
        linearmodel = smf.ols(formula=modified_formula, data=linear_grouped_agg)
    else:
        linearmodel = smf.wls(formula=modified_formula, data=linear_grouped_agg, weights=weights)
    results = linearmodel.fit()

    return results, modified_formula


def correct_brainage(inputs, params, correction_method, formula):
    brainage, modality, scanner, age = inputs

    # Create a boolean mask for rows where all values are zero
    rows_nomodality = tf.expand_dims(tf.reduce_all(tf.equal(modality, 0), axis=-1), axis=1)
    rows_noscanner = tf.expand_dims(tf.reduce_all(tf.equal(scanner, 0), axis=-1), axis=1)
    rows_nomodality_yesscanner = tf.logical_and(rows_nomodality, tf.logical_not(rows_noscanner))
    rows_yesmodality_noscanner = tf.logical_and(tf.logical_not(rows_nomodality), rows_noscanner)
    rows_nomodality_noscanner = tf.logical_and(rows_nomodality, rows_noscanner)
    rows_yesmodality_yesscanner = tf.logical_not(tf.logical_or(rows_nomodality, rows_noscanner))

    index_modality = formula.find("modality")
    index_scanner = formula.find("scanner")
    if index_modality != -1 and not index_scanner != -1:
        n = modality.shape[1]
        p = tf.transpose(tf.reshape(tf.constant(params, "float32"), [len(params) // n, n]))
        coeffs = tf.where(
            rows_nomodality, tf.reshape(tf.reduce_mean(p, axis=0), [1, -1]), tf.matmul(modality, p)
        )
    elif not index_modality != -1 and index_scanner != -1:
        m = scanner.shape[1]
        p = tf.transpose(tf.reshape(tf.constant(params, "float32"), [len(params) // m, m]))
        # select coefficients according to scanner
        coeffs = tf.where(
            rows_noscanner, tf.reshape(tf.reduce_mean(p, axis=0), [1, -1]), tf.matmul(scanner, p)
        )
    elif index_modality != -1 and index_scanner != -1:
        n = modality.shape[1]
        m = scanner.shape[1]
        p = tf.transpose(tf.reshape(tf.constant(params, "float32"), [len(params) // n, n]))
        p = tf.reshape(p, [p.shape[0], p.shape[1] // m, m])
        c_yesmodality_yesscanner = tf.einsum("ij,jkl,il->ik", modality, p, scanner)
        c_nomodality_noscanner = tf.reshape(tf.einsum("jkl->k", p) / (n * m), [1, -1])
        c_nomodality_yesscanner = tf.einsum("jkl,il->ik", p, scanner) / n
        c_yesmodality_noscanner = tf.einsum("ij,jkl->ik", modality, p) / m
        coeffs = (
            tf.where(rows_yesmodality_yesscanner, c_yesmodality_yesscanner, 0)
            + tf.where(rows_nomodality_noscanner, c_nomodality_noscanner, 0)
            + tf.where(rows_nomodality_yesscanner, c_nomodality_yesscanner, 0)
            + tf.where(rows_yesmodality_noscanner, c_yesmodality_noscanner, 0)
        )
    else:
        coeffs = tf.reshape(tf.constant(params, "float32"), [1, -1])

    if correction_method == "clear":
        if coeffs.shape[1] == 2:
            corrected_brainage = (
                brainage - tf.expand_dims(coeffs[:, 0], axis=-1)
            ) / tf.expand_dims(coeffs[:, 1], axis=-1)
        elif coeffs.shape[1] == 3:
            a = tf.expand_dims(coeffs[:, 2], axis=-1)
            b = tf.expand_dims(coeffs[:, 1], axis=-1)
            c = tf.expand_dims(coeffs[:, 0], axis=-1) - brainage
            D = tf.sqrt(tf.square(b) - 4 * a * c)
            D = tf.where(tf.math.is_nan(D), 0.0, D)
            solution0 = (-b + D) / (2 * a)
            solution1 = (-b - D) / (2 * a)
            # Compute the absolute differences
            diff1 = tf.abs(solution0 - brainage)
            diff2 = tf.abs(solution1 - brainage)
            # Create a mask that is True where tensor1 is closer to tensor3
            mask = tf.less(diff1, diff2)
            # Select values from tensor1 or tensor2 based on the mask
            corrected_brainage = tf.where(mask, solution0, solution1)

    elif correction_method == "cov":
        if coeffs.shape[1] == 2:
            prediction = (
                tf.expand_dims(c[:, 0], axis=-1) + tf.expand_dims(coeffs[:, 1], axis=-1) * age
            )
        elif coeffs.shape[1] == 3:
            prediction = (
                tf.expand_dims(coeffs[:, 0], axis=-1)
                + tf.expand_dims(coeffs[:, 1], axis=-1) * age
                + tf.expand_dims(coeffs[:, 2], axis=-1) * age**2
            )
        corrected_brainage = brainage + age - prediction

    return corrected_brainage


def group_by_ID(df):
    # Groups by UID
    if "corrected_brainage_slices" in df.columns:
        grouped = df.groupby(by="UID")
        grouped_agg = grouped.agg(
            {
                "corrected_brainage_slices": "median",
                "brainage_slices": "median",
                "age": "first",
                "modality": "first",
                "scanner": "first",
                "ID": "first",
            }
        )
        grouped_agg.rename(
            columns={"corrected_brainage_slices": "corrected_brainage"}, inplace=True
        )
        grouped_agg.rename(columns={"brainage_slices": "brainage"}, inplace=True)
        grouped_agg["corrected_PAD"] = grouped_agg["corrected_brainage"] - grouped_agg["age"]
        grouped_agg["PAD"] = grouped_agg["brainage"] - grouped_agg["age"]

        return grouped_agg
    else:
        grouped = df.groupby(by="UID")
        grouped_agg = grouped.agg(
            {
                "brainage_slices": "median",
                "age": "first",
                "modality": "first",
                "scanner": "first",
                "ID": "first",
            }
        )
        grouped_agg.rename(columns={"brainage_slices": "brainage"}, inplace=True)
        grouped_agg["PAD"] = grouped_agg["brainage"] - grouped_agg["age"]

        return grouped_agg


def cronbach_from_df(df, numitems=3):
    """
    Calculates the Cronbach's alpha of a data set df.

    Parameters:
    df: Data set. It has to be a DataFrame
    numitems: Number of items to be used when df contains missing values.
              Input None to allow missing values

    Returns:
    alpha: Cronbach's alpha.
    data: Matrix used to calculate Cronbach's alpha.

    Reference:
    Cronbach L J (1951): Coefficient alpha and the internal structure of
    tests. Psychometrika 16:297-333
    """

    if "corrected_PAD" in df.columns:
        pad_var = "corrected_PAD"
    else:
        pad_var = "PAD"

    df["repetition"] = df.groupby(["ID", "modality"]).cumcount() + 1

    # Get unique IDs and categories
    ids = df["ID"].unique()
    modalities = df["modality"].unique()
    repetitions = df["repetition"].unique()

    # Create new dataframe with all combinations of IDs and categories
    new_df = pd.DataFrame(
        {
            "ID": np.repeat(np.repeat(ids, len(modalities)), len(repetitions)),
            "modality": np.tile(np.repeat(modalities, len(repetitions)), len(ids)),
            "repetition": np.tile(repetitions, len(ids) * len(modalities)),
        }
    )

    # Merge new dataframe with original dataframe to get ages
    new_df = new_df.merge(
        df[["ID", "modality", "repetition", pad_var]],
        on=["ID", "modality", "repetition"],
        how="left",
    )

    # Sort new dataframe by ID and Category
    new_df = new_df.sort_values(by=["ID", "modality", "repetition"]).reset_index(drop=True)

    # Pivot table to create wide-format dataframe
    wide_df = new_df.pivot_table(index="ID", columns=["modality", "repetition"], values=pad_var)

    # Eliminate rows from wide_df where the number of non-missing values is less than numitems
    if numitems is not None:
        wide_df = wide_df.loc[wide_df.notna().sum(axis=1) >= numitems]

    non_nan_pct = wide_df.notna().sum() / len(df)
    wide_df = wide_df.loc[:, non_nan_pct >= 0.05]

    alpha = pg.cronbach_alpha(data=wide_df)
    # The line below is not to be used
    # because we need to drop out columns with too few non-NaN values
    # but I wanted to write it so we know it can be done
    # alpha = pg.cronbach_alpha(
    #     data=new_df, subject="ID", items=["modality", "repetition"], scores=pad_var
    # )

    return alpha


def myplots(
    df,
    figs,
    x="age",
    y=None,
    hue=None,
    hue_order=None,
    labels=None,
    labelsPAD=None,
    bysize=False,
):
    if y is None:
        y = ["corrected_brainage", "corrected_PAD"]
    if labels is None:
        labels = ["Chronological age", "Corrected predicted brain age"]
    if labelsPAD is None:
        labelsPAD = ["Participants", "Corrected brain-PAD"]


def create_model(
    dbnmodel, LAYERS_TO_TRAIN, nmodalities, nscanners, moderate, clone_last_layers=False, noise=0
):
    # Clone the DBN model to create a new model with the same architecture and weights
    model = clone_model(dbnmodel)
    model.set_weights(dbnmodel.get_weights())

    # save the last 3 layers
    dense0 = model.layers[-3]
    dropout0 = model.layers[-2]
    output0 = model.layers[-1]

    # Get the input layer of the cloned model (for the image data)
    image_input = model.input

    # Add noise to the input
    # noise1 = GaussianNoise(noise)(image_input)

    # Create an input layer for the categorical modalities
    modalities_input = layers.Input(shape=(nmodalities,), name="modalities_input")

    # Create an input layer for the categorical scanners
    scanners_input = layers.Input(shape=(nscanners,), name="scanners_input")

    # Create an input layer for the age
    age_input = layers.Input(shape=(1,), name="age_input")

    # Remove the last three layers from the cloned model
    model_output = model.layers[-LAYERS_TO_TRAIN - 1].output  # (noise1)

    if moderate:
        # Create an embedding layer to map the categorical data to a continuous representation
        embedding = layers.Embedding(input_dim=nmodalities, output_dim=nmodalities // 2)(
            modalities_input
        )
        embedding = layers.Flatten()(embedding)

        # Concatenate the image data and the embedded categorical data
        concat = layers.concatenate([model_output, embedding])
    else:
        concat = model_output

    # Pass the concatenated data through a dense layer with 1024 units and ReLU activation
    if clone_last_layers:
        dense1 = dense0(concat)
    else:
        dense1 = layers.Dense(1024, activation="relu")(concat)

    # Add a dropout layer with a rate of 0.5
    if clone_last_layers:
        drop = dropout0(dense1)
    else:
        drop = layers.Dropout(0.8)(dense1)

    # Add a final dense layer with 1 unit and linear activation to produce the output
    if clone_last_layers:
        output = output0(drop)
    else:
        output = layers.Dense(1, activation="linear")(drop)

    # Create a new model with the image and categorical inputs and the final output
    model = keras.Model(
        inputs=[image_input, modalities_input, scanners_input, age_input], outputs=output
    )

    # Print a summary of the new model
    # model.summary()

    return model
