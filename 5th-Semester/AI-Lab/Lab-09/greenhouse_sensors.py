import numpy as np

# --- (1) Generate reproducible readings ---
np.random.seed(7)
readings = np.random.randint(20, 41, size=(6, 3))
print(readings)

# Mean of each sensor (column)
means = readings.mean(axis=0)
print("Mean per sensor:", means)

# --- (2) Standardize each sensor with broadcasting ---
stds = readings.std(axis=0)
normalized = (readings - means) / stds          # (6,3) - (3,) broadcasts per column
print("Normalized column means:", normalized.mean(axis=0).round(10))
# round() suppresses tiny floating-point residue (~1e-16) so it prints 0.

# --- (3) Hour of peak per sensor ---
print("Hour of max per sensor:", readings.argmax(axis=0))

# --- (4) Append the mean row with vstack ---
with_means = np.vstack([readings, means])
print(with_means.shape)

# --- (5) Prove reproducibility ---
np.random.seed(7)
readings2 = np.random.randint(20, 41, size=(6, 3))
print("Reproducible:", np.array_equal(readings, readings2))

# (added on branch add-summary-row)
print("Overall hottest reading:", readings.max())