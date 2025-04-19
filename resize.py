import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

data_dir = '/scratch/l.peiwang/monash_new/'
target_shape = (96, 112, 96)

def resize_image(img_data, target_shape):
    factors = (
        target_shape[0] / img_data.shape[0],
        target_shape[1] / img_data.shape[1],
        target_shape[2] / img_data.shape[2]
    )
    resized_img = zoom(img_data, factors, order=3)
    return resized_img.astype(np.float32)

subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])

for subj_id in subjects:
    print(f'Resizing subject {subj_id}...')

    subj_dir = os.path.join(data_dir, subj_id)

    filenames = [
        f'{subj_id}_T1w_MNI.nii.gz',
        f'{subj_id}_fMRI_mean_MNI.nii.gz',
        f'{subj_id}_fPET_mean_MNI.nii.gz'
    ]

    for fname in filenames:
        filepath = os.path.join(subj_dir, fname)
        img_nib = nib.load(filepath)
        resized_data = resize_image(img_nib.get_fdata(), target_shape)

        # Overwrite directly in-place
        resized_img_nib = nib.Nifti1Image(resized_data, affine=img_nib.affine)
        nib.save(resized_img_nib, filepath)

print('✅ Images successfully resized in-place!')
