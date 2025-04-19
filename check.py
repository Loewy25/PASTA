import nibabel as nib
import os

# Modify the path and subject ID as needed
data_dir = '/scratch/l.peiwang/monash_new/'
subject_id = 'sub-01'

subject_dir = os.path.join(data_dir, subject_id)

# Define file paths clearly
t1_file = os.path.join(subject_dir, f'{subject_id}_T1w_MNI.nii.gz')
fmri_file = os.path.join(subject_dir, f'{subject_id}_fMRI_mean_MNI.nii.gz')
fpet_file = os.path.join(subject_dir, f'{subject_id}_fPET_mean_MNI.nii.gz')

# Load images with nibabel and print dimensions
t1_img = nib.load(t1_file)
fmri_img = nib.load(fmri_file)
fpet_img = nib.load(fpet_file)

print(f"T1 MRI shape: {t1_img.shape}")
print(f"fMRI mean shape: {fmri_img.shape}")
print(f"fPET mean shape: {fpet_img.shape}")
