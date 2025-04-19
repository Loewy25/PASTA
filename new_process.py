import os
import h5py
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# Paths setup
input_dir = '/scratch/l.peiwang/monash_new/'
output_dir = '/scratch/l.peiwang/hdf5_data_monash_fmri/'
os.makedirs(output_dir, exist_ok=True)

clinical_columns = ['age', 'gender', 'education', 'MMSE', 'ADAS-Cog-13', 'ApoE4']

# Random/dummy clinical data generator
def generate_dummy_tabular():
    return np.array([
        np.random.randint(20, 90),            # age
        np.random.randint(0, 2),              # gender (0 or 1)
        np.random.randint(8, 20),             # education years
        np.random.randint(20, 30),            # MMSE
        np.random.uniform(5, 25),             # ADAS-Cog-13
        np.random.randint(0, 3)               # ApoE4 genotype (0,1,2)
    ], dtype=np.float32)

# Get subject list
subjects = sorted([s for s in os.listdir(input_dir) if s.startswith('sub-')])

# Split into train, validation, and test (e.g., 70%, 15%, 15%)
train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

datasets = {
    'train.h5': train_subjects,
    'val.h5': val_subjects,
    'test.h5': test_subjects
}

# Process each set (train, val, test)
for filename, subj_list in datasets.items():
    all_tabular_data = []
    with h5py.File(os.path.join(output_dir, filename), 'w') as hdf5_file:
        print(f"Creating {filename} with {len(subj_list)} subjects...")
        
        for subj_id in subj_list:
            subj_dir = os.path.join(input_dir, subj_id)
            grp = hdf5_file.create_group(subj_id)

            # MRI/T1
            t1_data = nib.load(os.path.join(subj_dir, f'{subj_id}_T1w_MNI.nii.gz')).get_fdata().astype(np.float32)
            grp.create_dataset('MRI/T1', data=t1_data, compression='gzip')

            # PET/FDG (using fMRI data)
            fmri_data = nib.load(os.path.join(subj_dir, f'{subj_id}_fMRI_mean_MNI.nii.gz')).get_fdata().astype(np.float32)
            grp.create_dataset('PET/FDG', data=fmri_data, compression='gzip')

            # Tabular dummy data
            tabular_data = generate_dummy_tabular()
            grp.create_dataset('tabular', data=tabular_data)
            all_tabular_data.append(tabular_data)

            # Randomized or placeholder attributes
# Randomized or placeholder attributes (fixed encoding)
            grp.attrs['DX'] = np.random.choice(['CN', 'Dementia', 'MCI']).encode('utf-8')
            grp.attrs['RID'] = int(subj_id.split('-')[1])
            grp.attrs['VISCODE'] = np.random.choice(['bl', 'm06', 'm12']).encode('utf-8')


        # Calculate tabular statistics (mean, std)
        stats_grp = hdf5_file.create_group('stats/tabular')
        stats_grp.create_dataset('columns', data=np.array(clinical_columns, dtype='S'))

        tabular_array = np.stack(all_tabular_data)
        stats_grp.create_dataset('mean', data=np.mean(tabular_array, axis=0))
        stats_grp.create_dataset('stddev', data=np.std(tabular_array, axis=0))

print("🎉 Successfully created train, validation, and test HDF5 files!")
