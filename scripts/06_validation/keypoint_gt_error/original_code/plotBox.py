import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from violin import violin
import seaborn as sns

# Load data
# two = loadmat('TwoAnimals.mat')
single = loadmat('SingleAnimal_20.mat')
# three = loadmat('ThreeAnimals.mat')

# Concatenate 3D keypoint data
annotKp3ds = np.concatenate([
    # three['annotKp3ds'].reshape(18, 3, -1),
    # two['annotKp3ds'].reshape(18, 3, -1),
    single['annotKp3ds'].squeeze()
], axis=2)

decodedKp3ds = np.concatenate([
    # three['decodedKp3ds'].reshape(18, 3, -1),
    # two['decodedKp3ds'].reshape(18, 3, -1),
    single['decodedKp3ds'].squeeze()
], axis=2)

# Compute errors
gapdata = np.linalg.norm(annotKp3ds - decodedKp3ds, axis=1)

print(gapdata.shape)

# Replace NaN values with 0
gapdata = np.nan_to_num(gapdata, nan=0)

# Filter valid data (adjust the threshold to an appropriate value)
flgValid = np.nanmedian(gapdata, axis=0) < 1e6  # adjust threshold

fig, ax = plt.subplots(figsize=(10, 6))

ax.violinplot(gapdata[:, flgValid].T, 
              showmeans=False,  # do not mark the mean
              showmedians=True) # mark the median

# Axis labels
ax.set_xlabel('Keypoint Index')
ax.set_ylabel('Error Value')
ax.set_title('Violin Plot of Keypoint Errors Across Frames')

# Set the x-axis range up to 40
ax.set_ylim(0, 40)


# Set x-axis ticks to 1-18 and replace with keypoint names as needed
xticks = np.arange(1, gapdata.shape[0] + 1)
ax.set_xticks(xticks)
ax.set_xticklabels([f'KP{i}' for i in xticks], rotation=45)

plt.tight_layout()
plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set seaborn style
# sns.set(style="whitegrid")

# # gapdata is a NumPy array of shape (18,301)
# # seaborn's violinplot treats each column as a category, so transpose into a DataFrame
# df = pd.DataFrame(gapdata.T, columns=[f"KP{i+1}" for i in range(gapdata.shape[0])])

# plt.figure(figsize=(12, 6))
# sns.violinplot(data=df, 
#                inner="quartile",   # show interquartile range
#                scale="width"       # scale violin width to the number of data points
#               )

# plt.ylim(0, 40)
# plt.xticks(rotation=45)
# plt.xlabel("Keypoint Index")
# plt.ylabel("Error Value")
# plt.title("Seaborn Violin Plot of Keypoint Errors Across Frames")
# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style
sns.set(style="whitegrid")

# When gapdata is a NumPy array (18, 301)
# Create a DataFrame with one column per keypoint
df = pd.DataFrame(
    gapdata.T,
    columns=[f"KP{i+1}" for i in range(gapdata.shape[0])]
)

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    whis=1.5,        # default whisker length; up to 1.5x IQR
    showfliers=True, # show outliers as points
    notch=False      # True for notched boxes
)

plt.xticks(rotation=45)
plt.xlabel("Keypoint Index")
plt.ylabel("Error Value [mm]")
plt.title("Seaborn Box Plot of Keypoint Errors Across Frames")
plt.tight_layout()
plt.show()
