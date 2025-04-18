import os, nibabel as nib, numpy as np

input_dir = '/home/scratch/ds002898-1.4.2/'
output_dir = '/home/scratch/monash_new/'
os.makedirs(output_dir, exist_ok=True)

subjects = [s for s in os.listdir(input_dir) if s.startswith('sub-')]

for sub in subjects:
    print(f"Processing {sub}...")
    in_sub_dir = os.path.join(input_dir, sub)
    out_sub_dir = os.path.join(output_dir, sub)
    os.makedirs(out_sub_dir, exist_ok=True)

    # Load T1 MRI
    t1_path = os.path.join(in_sub_dir, 'anat', f'{sub}_T1w.nii.gz')
    t1_img = nib.load(t1_path)
    nib.save(t1_img, os.path.join(out_sub_dir, f'{sub}_T1w.nii.gz'))

    # Average fMRI across all runs and time frames
    fmri_runs = sorted([f for f in os.listdir(os.path.join(in_sub_dir, 'func')) if f.endswith('_bold.nii.gz')])
    fmri_averages = []
    for fmri_file in fmri_runs:
        fmri_img = nib.load(os.path.join(in_sub_dir, 'func', fmri_file))
        fmri_mean = np.mean(fmri_img.get_fdata(), axis=3)
        fmri_averages.append(fmri_mean)
    fmri_final = np.mean(fmri_averages, axis=0)
    nib.save(nib.Nifti1Image(fmri_final, t1_img.affine),
             os.path.join(out_sub_dir, f'{sub}_fMRI_mean.nii.gz'))

    # Average FDG-PET across time
    pet_img = nib.load(os.path.join(in_sub_dir, 'pet', f'{sub}_task-rest_trc-18FFDG_rec-acdyn_run-001_pet.nii.gz'))
    pet_mean = np.mean(pet_img.get_fdata(), axis=3)
    nib.save(nib.Nifti1Image(pet_mean, t1_img.affine),
             os.path.join(out_sub_dir, f'{sub}_fPET_mean.nii.gz'))

print("🎉 Averaging done!")
