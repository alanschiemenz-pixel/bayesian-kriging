# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../geochem.csv")

def scatter_plot(df, col, vmin=None, vmax=None):
    plt.scatter(df["x"], df["y"], c=df[col], cmap="YlOrRd", vmin=vmin, vmax=vmax)
    plt.colorbar(label=col)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# %% # scale it

scaler = StandardScaler()

cols_to_scale = [c for c in df.columns if c not in ("x", "y", "near_drill")]
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
# %%  summary stats of scaled
df_scaled.describe()

# %% plot it
for col in cols_to_scale:
    scatter_plot(df_scaled,col)

# Cu has anomalies that are uncorrelated in top left

# %% flag anomalies
df_scaled["anomaly"] = (df_scaled[cols_to_scale] > 2.5).any(axis=1).astype(int)
df_scaled["anomaly_score"] = (df_scaled[cols_to_scale] > 2.5).sum(axis=1)

# %% plot anomaly
scatter_plot(df_scaled,'anomaly_score')


# %% plot Cu
scatter_plot(df_scaled,'Cu',vmin=0,vmax=5)
# spatial halo around (7000,7000)

# %% cross-correlation
import seaborn as sns

corr = df_scaled[cols_to_scale].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Cross-correlation matrix")
plt.tight_layout()
plt.show()
# high correlations between Cu,Mo,Au,As

# %% isolation forest
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df_scaled["iso_anomaly"] = iso.fit_predict(df_scaled[cols_to_scale])   # -1 = anomaly, 1 = normal
df_scaled["iso_score"]   = iso.decision_function(df_scaled[cols_to_scale])  # lower = more anomalous

print(df_scaled["iso_anomaly"].value_counts())
scatter_plot(df_scaled, "iso_anomaly")

# %% halo effect on Cu (point data, KDTree radius search)
from scipy.spatial import KDTree

RADIUS = 500   # metres -- tune to your deposit scale

tree = KDTree(df_scaled[["x", "y"]].values)
cu   = df_scaled["Cu"].values

halo = np.zeros(len(df_scaled))
for i, pt in enumerate(df_scaled[["x", "y"]].values):
    neighbour_idx = tree.query_ball_point(pt, r=RADIUS)
    neighbour_idx = [j for j in neighbour_idx if j != i]   # exclude self
    if neighbour_idx:
        halo[i] = cu[neighbour_idx].mean()
    else:
        halo[i] = cu[i]   # no neighbours: use own value

df_scaled["Cu_halo"] = halo
# Positive values mean neighbours are richer than you -> you are in a halo zone
df_scaled["Cu_halo_contrast"] = halo - cu

scatter_plot(df_scaled, "Cu_halo")
scatter_plot(df_scaled, "Cu_halo_contrast")
