#%%
import numpy as np
import pandas as pd
# %%

np.random.seed(42)
n = 800
x = np.random.uniform(0, 10000, n)
y = np.random.uniform(0, 10000, n)

# %%
# Background populations
cu = np.random.lognormal(mean=3.5, sigma=0.8, size=n)
mo = cu * np.random.uniform(0.01, 0.05, size=n) + np.random.lognormal(1.5, 0.5, n)
au = mo * np.random.uniform(0.001, 0.01, size=n) + np.random.lognormal(0.1, 0.3, n)
as_ = cu * np.random.uniform(0.1, 0.3, size=n) + np.random.lognormal(2.0, 0.6, n)
pb = np.random.lognormal(2.5, 0.7, size=n)
zn = np.random.lognormal(3.0, 0.6, size=n)
mn = np.random.lognormal(5.5, 0.5, size=n)
fe = np.random.lognormal(8.0, 0.4, size=n)

# Inject a real anomaly — spatial cluster near (7000, 7000)
anomaly_idx = np.where((np.abs(x - 7000) < 600) & (np.abs(y - 7000) < 600))[0]
cu[anomaly_idx] *= np.random.uniform(8, 15, size=len(anomaly_idx))
mo[anomaly_idx] *= np.random.uniform(6, 12, size=len(anomaly_idx))
au[anomaly_idx] *= np.random.uniform(5, 10, size=len(anomaly_idx))
as_[anomaly_idx] *= np.random.uniform(3, 6, size=len(anomaly_idx))

# Inject data errors — isolated spikes, no halo, single element
error_idx = np.random.choice(n, 5, replace=False)
cu[error_idx] *= np.random.uniform(20, 50, size=5)

# Three drill holes in northeast corner — already sampled, low surprise value
near_drill = ((x > 8000) & (y > 8000)).astype(int)

df = pd.DataFrame({
    'x': x, 'y': y,
    'Cu': cu, 'Mo': mo, 'Au': au, 'As': as_,
    'Pb': pb, 'Zn': zn, 'Mn': mn, 'Fe': fe,
    'near_drill': near_drill
})

df.to_csv('geochem.csv', index=False)
print(df.shape)
df.head()