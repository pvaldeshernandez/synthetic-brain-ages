import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from patsy import dmatrices, Treatment

from utils import export_test_results

# Read data
df0 = pd.read_csv(
    "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results_dbn/test_predictions_dbn.csv"
)
df0["modality"] = df0["modality"].replace("MPRAGE", "MPRAGE0")
df0 = df0.assign(
    corrected_brainage=np.random.rand(len(df0)), corrected_PAD=np.random.rand(len(df0))
)
df1 = pd.read_csv(
    "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/test_predictions.csv"
)
df = pd.concat([df1, df0[df0["modality"].isin(["MPRAGE0"])]], ignore_index=False)
df = df.reset_index(drop=True)

mask = df["modality"].isin(["T1wGRE-SR"])
df = df[~mask]

# reorder modalities
modalities = [
    "MPRAGE",
    "MPRAGE-SR",
    "T1w-SR",
    "T2w-SR",
    "T1wFLAIR-SR",
    "T2wFLAIR-SR",
    "IR-SR",
    "T2wGRE-SR",
    "MPRAGE0",
]
df["modality"] = pd.Categorical(df["modality"], categories=modalities, ordered=True)

# Calculate the absolute value of the corrected_PAD
df["PAD"] = df["PAD"].abs()
df["corrected_PAD"] = df["corrected_PAD"].abs()

# Convert ID to category
df["ID"] = df["ID"].astype("category")

for pad_var in ["PAD", "corrected_PAD"]:
    if pad_var == "corrected_PAD":
        df = df.copy()
        df = df[~df["modality"].isin(["MPRAGE0"])]
        df = df.reset_index(drop=True)
        df["modality"] = pd.Categorical(df["modality"], categories=modalities[:-1], ordered=True)

    # Number of modalities
    nmodalities = df["modality"].nunique()

    # Get indices excluding T2w-GRE and/or MPRAGE0
    idx = np.r_[0:nmodalities]
    if pad_var == "corrected_PAD":
        idx = np.delete(idx, [modalities[:-1].index("T2wGRE-SR")])
    else:
        idx = np.delete(idx, [modalities.index("T2wGRE-SR"), modalities.index("MPRAGE0")])

    # Specify the model formula
    formula = pad_var + " ~ modality + ID"

    # Fit mixed effects model
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["ID"],
    )
    result = model.fit(maxiter=10000)

    # Print summary of model fit
    # print(result.summary(alpha=0.05 / 1))

    # Convert summary table to DataFrame
    summary_df = result.summary(alpha=0.05 / 1).tables[1]

    # Write DataFrame to CSV file
    summary_df.to_csv(
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/summary-{pad_var}.csv"
    )

    # ANOVA across modalities
    anova_contrast = np.eye(nmodalities - 1, result.params.shape[0] - 1)
    anova_contrast = np.hstack((np.zeros([nmodalities - 1, 1]), anova_contrast))
    export_test_results(
        result,
        anova_contrast,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/anova-{pad_var}.csv",
        test_type="f",
    )

    # Comparison with MPRAGE
    anova_contrast = anova_contrast[:, :-1]
    avg_row = np.mean(anova_contrast[idx, :], axis=0)
    anova_contrast = np.vstack((anova_contrast, avg_row))
    t_test = export_test_results(
        result,
        anova_contrast,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/comparison-{pad_var}.csv",
        test_type="t",
        alpha=0.05 / anova_contrast.shape[0],
    )

    # MAEs
    nids = result.params.shape[0] - nmodalities + 1
    mae_contrast0 = np.ones((nmodalities, 1))
    mae_contrast1 = np.diag(np.insert(np.ones(nmodalities - 1), 0, 0))
    mae_contrast2 = np.ones((nmodalities, result.params.shape[0] - nmodalities - 1)) / nids
    mae_contrast = np.hstack((mae_contrast0, mae_contrast1[:, 1:], mae_contrast2))
    avg_row = np.mean(mae_contrast[idx, :], axis=0)
    mae_contrast = np.vstack((mae_contrast, avg_row))
    export_test_results(
        result,
        mae_contrast,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/maes-{pad_var}.csv",
        test_type="t",
        alpha=0.05 / mae_contrast.shape[0],
    )

    # Test differences in slopes
    # Specify the model formula
    formula = pad_var + " ~ modality * age - age"

    # df2 = df[df["modality"] != "T2wGRE-SR"].copy()
    # df2["modality"] = df2["modality"].cat.remove_categories("T2wGRE-SR")

    # Fit mixed effects model
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["ID"],
    )
    result = model.fit()

    # ANOVA across pairs
    interaction_terms = [param for param in result.params.index if ":age" in param]

    # Create a contrast matrix
    anova_contrast = np.zeros((len(interaction_terms) - 1, len(result.params)))
    for i in range(len(interaction_terms) - 1):
        anova_contrast[i, result.params.index.get_loc(interaction_terms[i])] = -1
        anova_contrast[i, result.params.index.get_loc(interaction_terms[i + 1])] = 1

    export_test_results(
        result,
        anova_contrast,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/anova-slope-{pad_var}.csv",
        test_type="f",
    )

    # Perform pairwise t-tests
    contrast_matrix = np.zeros(
        (len(interaction_terms) * (len(interaction_terms) - 1) // 2, len(result.params))
    )
    pair_names = []
    k = 0
    for i in range(len(interaction_terms)):
        for j in range(i + 1, len(interaction_terms)):
            contrast_matrix[k, result.params.index.get_loc(interaction_terms[i])] = -1
            contrast_matrix[k, result.params.index.get_loc(interaction_terms[j])] = 1
            pair_name = (
                f"{interaction_terms[i]} vs {interaction_terms[j]}".replace("modality", "")
                .replace(":age", "")
                .replace("[", "")
                .replace("]", "")
            )
            pair_names.append(pair_name)
            k += 1
    contrast_matrix = contrast_matrix[:, :-1]

    export_test_results(
        result,
        contrast_matrix,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/anova-slope-pairwise-{pad_var}.csv",
        test_type="t",
        names=pair_names,
        alpha=0.05 / contrast_matrix.shape[0],
    )

    # Test if different from zero
    anova_contrast = np.eye(len(result.params))
    # Select only the rows corresponding to the interaction terms
    interaction_rows = [i for i, param in enumerate(result.params.index) if ":age" in param]
    anova_contrast = anova_contrast[interaction_rows, :]

    export_test_results(
        result,
        anova_contrast,
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/anova-slope_zero-{pad_var}.csv",
        test_type="f",
    )

    anova_contrast = np.vstack((anova_contrast, np.mean(anova_contrast[idx, :], axis=0)))
    export_test_results(
        result,
        anova_contrast[:, :-1],
        f"/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results/anova-slope_zero-each-{pad_var}.csv",
        test_type="t",
        names=df["modality"].cat.categories.tolist() + ["all"],
        alpha=0.05 / anova_contrast.shape[0],
    )
