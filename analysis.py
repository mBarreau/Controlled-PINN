# %%
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["text.usetex"] = True

# %% Accuracy plot
base = Path("output/accuracy")
rows = []

for training in ["vanilla", "primal-dual"]:
    for category in ["match", "mismatch"]:
        is_match = category == "match"
        folder = base / training / category

        for file in folder.glob("*.csv"):
            df = pd.read_csv(file, header=None, names=["data"])
            value = float(file.stem.replace("_", "."))  # file name without .csv
            for v in df["data"]:
                rows.append(
                    {
                        "training": training,
                        "match": is_match,
                        "forgetting_factor": np.round(value, 1),
                        "normalized_l2_error": v * 100,
                    }
                )

final_df = pd.DataFrame(rows)

forgetting_factors = final_df["forgetting_factor"].unique()
vanilla_expanded = []
for _, row in final_df.query("training == 'vanilla'").iterrows():
    for ff in forgetting_factors:
        new_row = row.copy()
        new_row["forgetting_factor"] = ff
        vanilla_expanded.append(new_row)

vanilla_expanded_df = pd.DataFrame(vanilla_expanded)

final_df = pd.concat(
    [final_df.query("training == 'primal-dual'"), vanilla_expanded_df],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=final_df,
    x="forgetting_factor",
    y="normalized_l2_error",
    hue="match",
    style="training",
    marker="o",
    dashes=True,
    ax=ax,
    errorbar=("ci", 95),
    legend=False,
)
plt.grid()
ax.set_xlim(0.0, None)
ax.set_xlabel(r"Forgetting Factor")
ax.set_ylabel(r"Normalized $L_2$ Error (\%)")

custom_handles = [
    # match: True/False → "Match"/"Mismatch"
    Line2D([0], [0], color="C0", marker="o", linestyle="-", label="Mismatch"),
    Line2D([0], [0], color="C1", marker="o", linestyle="-", label="Match"),
    # training: Vanilla/Primal-Dual → "Vanilla"/"Integrator" (different linestyle)
    Line2D([0], [0], color="black", linestyle="--", label="Vanilla"),
    Line2D([0], [0], color="black", linestyle="-", label="Controlled"),
]
ax.legend(
    handles=custom_handles,
    ncol=2,
)
plt.savefig("output/forgetting_factor_analysis.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %% Weight plot
base = Path("output/weights")
rows = []

for category in ["match", "mismatch"]:
    is_match = category == "match"
    folder = base / "primal-dual" / category

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file, header=None, names=["seed", "data"])
        seed_ = df["seed"].iloc[0]
        epoch = 1
        value = float(file.stem.replace("_", "."))  # file name without .csv
        for v in df.iterrows():
            if v[1]["seed"] != seed_:
                epoch = 1
                seed_ = v[1]["seed"]
            rows.append(
                {
                    "seed": v[1]["seed"],
                    "epoch": epoch,
                    "weight": v[1]["data"],
                    "match": is_match,
                    "forgetting_factor": np.round(value, 1),
                }
            )
            epoch += 1

final_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=final_df,
    x="epoch",
    y="weight",
    hue="match",
    style="forgetting_factor",
    dashes=True,
    ax=ax,
    legend=False,
)
custom_handles = [
    # match: True/False → "Match"/"Mismatch"
    Line2D([0], [0], color="C0", linestyle="-", label="Mismatch"),
    Line2D([0], [0], color="C1", linestyle="-", label="Match"),
    # training: Vanilla/Primal-Dual → "Vanilla"/"Integrator" (different linestyle)
    Line2D([0], [0], color="black", linestyle="-", label="Integrator"),
    Line2D([0], [0], color="black", linestyle="--", label="Leaky Integrator"),
]
ax.legend(
    handles=custom_handles,
    ncol=2,
)
plt.grid()
ax.set_xlim(0.0, None)
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Weight value")
plt.savefig("output/weight_analysis.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
