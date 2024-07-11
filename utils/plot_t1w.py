import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file
nifti_file = '../data/Structural Preprocessed for 7T (1.6mm:59k mesh)/100610/MNINonLinear/T1w_restore.1.60.nii'
img = nib.load(nifti_file)
data = img.get_fdata()

# Plotting a slice from the middle of the volume
slice_index = data.shape[2] // 2  # Taking the middle slice along the z-axis
plt.imshow(data[:, :, slice_index], cmap='gray')
plt.colorbar()
plt.title('Middle Slice of T1w_restore.1.60.nii')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
