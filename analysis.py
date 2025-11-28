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

for category in ["match", "mismatch"]:
    is_match = category == "match"
    folder = base / category

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file, header=None, names=["data"])
        value = float(file.stem.replace("_", "."))
        for v in df["data"]:
            rows.append(
                {
                    "match": is_match,
                    "forgetting_factor": np.round(value, 1),  # file name without .csv
                    "normalized_l2_error": v * 100,
                }
            )

final_df = pd.DataFrame(rows)

# %%
plt.figure()
sns.lineplot(data=final_df, x="forgetting_factor", y="normalized_l2_error", hue="match")
plt.grid()
plt.xlabel(r"Forgetting Factor")
plt.ylabel(r"Normalized $L_2$ Error (\%)")
plt.show()

# %%
