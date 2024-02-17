import pickle
import numpy as np
import pandas as pd
from statsmodels.tools.eval_measures import aic
import seaborn as sns
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, std=0.1):
    """
    Adds Gaussian noise to an image
    """
    noisy_image = image + np.random.normal(mean, std, image.shape)
    # clip values between 0 and 1
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def calculate_aic(linearmodel, df, variable):
    # Extract the model parameters
    k = len(linearmodel.params)

    # Make predictions on the new data
    y_pred = linearmodel.predict(df)

    # Calculate the residual sum of squares
    rss = np.sum((df[variable] - y_pred) ** 2)

    # Calculate the AIC
    aic_value = aic(rss, df.shape[0], k)

    return aic_value


def calculate_aic_corr(linearmodel, df):
    # Extract the model parameters
    k = len(linearmodel.params)

    # Calculate the residual sum of squares
    rss = np.sum((df["age"] - df["corrected_brainage"]) ** 2)

    # Calculate the AIC
    aic_value = aic(rss, df.shape[0], k)

    return aic_value


def save_state(filename, state):
    with open(filename, "wb") as f:
        pickle.dump(state, f)


def load_state(filename):
    try:
        with open(filename, "rb") as f:
            state = pickle.load(f)
        return state
    except FileNotFoundError:
        return {}


def export_test_results(result, contrast_matrix, filename, test_type="t", names=None, alpha=0.05):
    # Perform the specified test
    if test_type == "f":
        f_test = result.f_test(contrast_matrix)
        df = pd.DataFrame(
            {
                "F-statistic": [f_test.fvalue],
                "P-value": [f_test.pvalue],
                "DF numerator": [f_test.df_num],
                "DF denominator": [f_test.df_denom],
            }
        )
        df = df.round(2)
        test = f_test
    elif test_type == "t":
        t_test = result.t_test(contrast_matrix)
        df = pd.DataFrame(
            {
                "Contrast": range(1, len(t_test.effect) + 1),
                "Effect": t_test.effect,
                "Standard error": t_test.sd,
                "T-statistic": t_test.tvalue,
                "P-value": t_test.pvalue,
                "Lower CI": t_test.conf_int(alpha)[:, 0],
                "Upper CI": t_test.conf_int(alpha)[:, 1],
            }
        )
        df = df.round(2)
        # Write the CI interval
        df["CI"] = df.apply(lambda row: f"[{row['Lower CI']}, {row['Upper CI']}]", axis=1)
        if names is not None:
            df.index = names
        test = t_test
    else:
        raise ValueError(f"Invalid test type: {test_type}")

    # Export the DataFrame to a CSV file
    if names is not None:
        df.to_csv(filename, index=True, index_label="pairs")
    else:
        df.to_csv(filename, index=False)

    return test


def plot_with_sizes(df, x, y, hue, hue_order, palette):
    # Calculate the number of points for each category
    counts = df[hue].value_counts()

    # Create a new order based on the counts
    new_order = counts.index.tolist()[::-1]

    # Create a new palette and size_dict based on the new order
    new_palette = {category: palette[category] for category in new_order}
    size_dict = {category: 10 * (len(hue_order) - size) for size, category in enumerate(new_order)}
    alpha_dict = {
        category: (len(hue_order) - (alpha) / 2) / (len(hue_order))
        for alpha, category in enumerate(new_order)
    }
    # size_dict = {category: counts[category]/20 for category in new_order}

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Loop over each category
    for category in new_order:
        # Filter the dataframe for the current category
        df_filtered = df[df[hue] == category]

        # Get the size for the current category
        size = size_dict[category]

        # Get the transparency for the current category
        alpha = alpha_dict[category]

        # Get the color for the current category from the palette
        color = new_palette[category]

        # Plot with regplot on the same axes
        sns.regplot(
            data=df_filtered,
            x=x,
            y=y,
            scatter_kws={"s": size, "color": color, "alpha": alpha},
            line_kws={"linewidth": 3, "linestyle": "--", "color": color},
            ax=ax,
            fit_reg=True,
            ci=None,  # Turn off confidence interval
        )
    # Adjust figure to make room for legend
    plt.subplots_adjust(right=0.7)

    # Add a legend manually
    for category in hue_order:
        plt.scatter([], [], s=size_dict[category], color=new_palette[category], label=category)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 0.5), loc="center left")
    return fig, ax
