# %%
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"] = True

# %% Import data
base = Path("output")
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

# %%
plt.figure()
sns.lineplot(
    data=final_df.query("training == 'primal-dual'"),
    x="forgetting_factor",
    y="normalized_l2_error",
    hue="match",
)
plt.grid()
plt.xlim(0.0, None)
plt.xlabel(r"Forgetting Factor")
plt.ylabel(r"Normalized $L_2$ Error (\%)")
plt.savefig("output/forgetting_factor_analysis.eps", dpi=300)
plt.show()

# %%
vanilla_df = (
    final_df.query("training == 'vanilla'")
    .groupby("match")
    .agg({"normalized_l2_error": ["mean", "std"]})
)
print(f"Vanilla PINN: {vanilla_df}")

pd_df = (
    final_df.query("training == 'primal-dual' and forgetting_factor == 0")
    .groupby("match")
    .agg({"normalized_l2_error": ["mean", "std"]})
)
print(f"Primal-Dual PINN: {pd_df}")

# %%
