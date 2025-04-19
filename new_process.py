import os
import numpy as np
import nibabel as nib
import h5py
from sklearn.model_selection import train_test_split

# Paths to your preprocessed data (adjust as needed)
input_dir = '/scratch/l.peiwang/monash_new/'
output_dir = '/scratch/l.peiwang/hdf5_data_monash_fpet/'
os.makedirs(output_dir, exist_ok=True)

clinical_columns = ['age', 'gender', 'education', 'MMSE', 'ADAS-Cog-13', 'ApoE4']

# Generate dummy clinical/tabular data (replace if real data is available)
def generate_dummy_tabular():
    return np.array([
        np.random.randint(55, 85),            # age
        np.random.randint(0, 2),              # gender (0 or 1)
        np.random.randint(8, 20),             # education
        np.random.randint(25, 30),            # MMSE
        np.random.uniform(5, 20),             # ADAS-Cog-13
        np.random.randint(0, 3)               # ApoE4
    ], dtype=np.float32)

# Get all subjects
subjects = sorted([s for s in os.listdir(input_dir) if s.startswith('sub-')])

# Split data clearly: 70% train, 15% val, 15% test
train_subj, tmp_subj = train_test_split(subjects, test_size=0.3, random_state=42)
val_subj, test_subj = train_test_split(tmp_subj, test_size=0.5, random_state=42)

datasets = {
    'train.h5': train_subj,
    'val.h5': val_subj,
    'test.h5': test_subj
}

# Helper function for attributes (clearly encoded as bytes)
def random_attribute(choice_list):
    return np.random.choice(choice_list).encode('utf-8')

# Generate HDF5 datasets
for filename, subj_list in datasets.items():
    tabular_data_list = []
    with h5py.File(os.path.join(output_dir, filename), 'w') as hdf5_file:
        print(f"Creating {filename}... ({len(subj_list)} subjects)")
        
        for subj_id in subj_list:
            subj_path = os.path.join(input_dir, subj_id)
            grp = hdf5_file.create_group(subj_id)

            # Load Structural MRI (T1)
            t1_img = nib.load(os.path.join(subj_path, f'{subj_id}_T1w_MNI.nii.gz'))
            t1_data = t1_img.get_fdata().astype(np.float32)
            grp.create_group('MRI/T1').create_dataset('data', data=t1_data, compression='gzip')

            # Load fMRI mean data (stored under PET/FDG as per your requirement)
            fmri_img = nib.load(os.path.join(subj_path, f'{subj_id}_fPET_mean_MNI.nii.gz'))
            fmri_data = fmri_img.get_fdata().astype(np.float32)
            grp.create_group('PET/FDG').create_dataset('data', data=fmri_data, compression='gzip')

            # Tabular data (dummy/random)
            tab_data = generate_dummy_tabular()
            grp.create_dataset('tabular', data=tab_data)
            tabular_data_list.append(tab_data)

            # Randomized attributes (clearly encoded strings)
            grp.attrs['DX'] = random_attribute(['CN', 'Dementia'])
            grp.attrs['RID'] = int(subj_id.split('-')[1])
            grp.attrs['VISCODE'] = random_attribute(['bl', 'm06', 'm12'])

        # Tabular statistics (mean, stddev across dataset)
        tabular_data_array = np.stack(tabular_data_list, axis=0)
        stats_grp = hdf5_file.create_group('stats/tabular')
        stats_grp.create_dataset('columns', data=np.array(clinical_columns, dtype='S'))
        stats_grp.create_dataset('mean', data=np.mean(tabular_data_array, axis=0))
        stats_grp.create_dataset('stddev', data=np.std(tabular_data_array, axis=0))

print("✅ HDF5 datasets (train, val, test) successfully created!")
