import os
import shutil
import pickle
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# --- Helper functions ---
def plotMarmo3D(kp3d, ax):
    """
    Plot 3D keypoints (replacement for MATLAB's plotMarmo3D).
    kp3d is assumed to be an (N,3) numpy array; here it is drawn as a scatter plot of points.
    Add drawing of skeleton lines etc. as needed.
    """
    # Plot keypoints as a scatter plot
    ax.scatter(kp3d[:, 0], kp3d[:, 1], kp3d[:, 2], s=20)
    # Example: connect with lines if skeleton info is available (placeholder example below)
    # e.g. skeleton = [(0,1), (1,2), ...]
    # for i, j in skeleton:
    #     ax.plot([kp3d[i,0], kp3d[j,0]],
    #             [kp3d[i,1], kp3d[j,1]],
    #             [kp3d[i,2], kp3d[j,2]], 'k-')
    return

kps=20

# --- Load CSV ---
# Read the CSV file and use the first column as the index
dtbl = pd.read_csv('./labeledGT/valbin1_nanim1_nfrm100/framedata.csv', index_col=0)

# Get the list of sessions (using the VideoSession column)
sessions = dtbl['VideoSession'].unique()
print(sessions)

# result_path = '../../../pose_det_result/3d_v0p8_dark_fix_20_all_det'
# sessions = glob.glob('*',root_dir=result_path)
# print(sessions)

# exit()

# --- Prepare figure ---
# (Create an 8x8 grid of subplots, approximating tight_subplot)
fig_main, axs_main = plt.subplots(8, 8, figsize=(12, 12), subplot_kw={'projection': '3d'})
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
axs_main = axs_main.flatten()
print(f"axs_main: {axs_main}")
# --- Storage for keypoint data ---
allgaps_list = []      # per-frame gaps (norms) of the 18 points
annotKp3ds_list = []   # annotated keypoints from JSON ((18,3) array per frame)
decodedKp3ds_list = [] # computed keypoints from pickle (same as above)

cnt = 0  # frame counter

# --- Session loop ---
# MATLAB uses for ises=1:10, so process the first 10
for session in sessions[:10]:
    print("Session:", session)
    # Extract only the rows of this session
    currdtbl = dtbl[dtbl['VideoSession'] == session]

    # --- Prepare pickle file ---
    dirPickleRoot = '../../results/3d_v0p8_dark_fix_20_for_3d_eval'
    # dirPickleRoot = result_path
    picklefile = os.path.join(dirPickleRoot, session, 'kp3d_fxdJointLen.pickle')
    if not os.path.exists(picklefile):
        dirPickleRoot2 = '../../results/3d_v0p8_dark_fix_20_for_3d_eval'
        # dirPickleRoot2 = result_path
        picklefile2 = os.path.join(dirPickleRoot2, session, 'kp3d_fxdJointLen.pickle')
        os.makedirs(os.path.join(dirPickleRoot, session), exist_ok=True)
        shutil.copyfile(picklefile2, picklefile)
    
    # --- Load data from pickle file ---
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    # Convert the 'kp3d' key in the pickle file to a NumPy array (expected shape: (nAnim, nFrame, nKeypoints, 3))
    kp3d = np.array(data['kp3d'], dtype=float)
    # print(kp3d.shape)
    
    # --- Process each frame ---
    # Process each row of the DataFrame
    for i in range(len(currdtbl)):
        cnt += 1
        ianim = 0  # was 1 in MATLAB, but Python is 0-based
        
        # MATLAB: currfrm = round(mean(cell2mat(table2cell(currdtbl(ifrm,2:end)))));
        # Convert values from the 2nd column of dtbl onward to numeric, average and round
        row = currdtbl.iloc[i]
        numeric_vals = row.iloc[1:].astype(float)
        currfrm = round(numeric_vals.mean())
        # In Kaneko-san's early data, the first frame was skipped;
        currfrm = currfrm - 24
        # Adjust to 0-index in Python (if currfrm is a 1-based frame number)
        # py_frame_index = currfrm - 1
        
        # Extract keypoints of the target frame from kp3d
        currKp3d = np.squeeze(kp3d[ianim, currfrm, :, :])
        
        # --- Load annotation data from JSON file ---
        # MATLAB gets the frame name from the Row property, so use the DataFrame index
        frmName = currdtbl.index[i]
        json_file = os.path.join('./labeledGT/valbin1_nanim1_nfrm100', f'{frmName}.json')
        with open(json_file, 'r') as jf:
            jsondata = json.load(jf)
        # Convert 'keypoints_3d' in the JSON to a NumPy array.
        # MATLAB uses ianim, 1:18, :, so in Python take ianim=0 and slice indices 0-17
        if kps == 19:
            annotKp3d = np.array(jsondata['keypoints_3d'])[ianim, np.r_[0:18, 19], :]
        elif kps == 20 :
            annotKp3d = np.array(jsondata['keypoints_3d'])[ianim, np.r_[0:20], :]
        else :
            annotKp3d = np.array(jsondata['keypoints_3d'])[ianim, np.r_[0:18], :]

        # # Create a new figure for 3D plotting
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot annotKp3d (blue)
        # ax.scatter(annotKp3d[:, 0], annotKp3d[:, 1], annotKp3d[:, 2], c='blue', marker='o', label='Ground Truth')

        # # Plot currKp3d (red) 
        # ax.scatter(currKp3d[:, 0], currKp3d[:, 1], currKp3d[:, 2], c='red', marker='o', label='Prediction')

        # # Set axis labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y') 
        # ax.set_zlabel('Z')

        # # Show legend
        # ax.legend()

        # # Show plot
        # plt.show()
        
        
        # --- Plotting ---
        if cnt <= len(axs_main):  # plot only while within the number of available subplots
            ax = axs_main[cnt-1]
            # Plot annotated and computed keypoints
            plotMarmo3D(annotKp3d, ax)
            plotMarmo3D(currKp3d, ax)
            # Set view angle
            ax.view_init(elev=10, azim=20)
            # Set title
            ax.set_title(f'Frame {cnt}')
        
        # --- Compute the gap (difference) between the two sets of keypoints ---
        gap = np.linalg.norm(annotKp3d - currKp3d, axis=1)  # 3D Euclidean norm per keypoint
        allgaps_list.append(gap)
        
        # --- Store keypoints for each frame ---
        annotKp3ds_list.append(annotKp3d)
        decodedKp3ds_list.append(currKp3d)

# --- Convert lists to arrays ---
# Each list is assumed to hold an (18, 3) array per frame
annotKp3ds = np.stack(annotKp3ds_list, axis=-1)     # example result shape: (18, 3, num_frames)
decodedKp3ds = np.stack(decodedKp3ds_list, axis=-1)
allgaps = np.stack(allgaps_list, axis=0)               # example result shape: (num_frames, 18)

print(annotKp3ds.shape)
print(decodedKp3ds.shape)


# # --- Save results in MATLAB format ---
savemat('SingleAnimal_20_test.mat', {'annotKp3ds': annotKp3ds, 'decodedKp3ds': decodedKp3ds})

# # --- Show legend ---
# plt.legend()

# # Show the first figure
# plt.figure(fig_main.number)
# plt.show()

# # --- Plot 100 frames (second loop) ---
# # Here ianim=0 (assumed to be the only animal)
# fig2, axes2 = plt.subplots(10, 10, figsize=(15, 15), subplot_kw={'projection': '3d'})
# axes2 = axes2.flatten()
# num_frames = 100
# for ifrm in range(num_frames):
#     # Plot only if the frame exists (otherwise hide the subplot)
#     if ifrm < annotKp3ds.shape[-1]:
#         annotKp3d = np.squeeze(annotKp3ds[:, :, ifrm])
#         currKp3d = np.squeeze(decodedKp3ds[:, :, ifrm])
#         ax = axes2[ifrm]
#         # Plot annotated and computed keypoints
#         plotMarmo3D(annotKp3d, ax)
#         plotMarmo3D(currKp3d, ax)
#         # Set view angle (equivalent to MATLAB's view([10 20]))
#         ax.view_init(elev=10, azim=20)
#     else:
#         axes2[ifrm].axis('off')

# plt.show()
