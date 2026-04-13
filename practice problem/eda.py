# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../geochem.csv")



# %% # scale it

scaler = StandardScaler()

cols_to_scale = [c for c in df.columns if c not in ("x", "y", "near_drill")]
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
# %%  summary stats of scaled
df_scaled.describe()

# %% plot it
for col in cols_to_scale:
    plt.scatter(df_scaled["x"], df_scaled["y"], c=df_scaled[col], cmap="YlOrRd")
    plt.colorbar(label=col)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Cu has anomalies that are uncorrelated in top left

# %% flag anomalies
df_scaled["anomaly"] = (df_scaled[cols_to_scale] > 2.5).any(axis=1).astype(int)