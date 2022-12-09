from pathlib import Path
from utils.pre_preocessing_heatmap_utils import *

print('Preprocessing training data')

training_v19_raw = Path('VerSe19/dataset-verse19training/rawdata/')
training_v20_raw = Path('VerSe20/dataset-01training/rawdata/')


training_v19_raw = [f for f in training_v19_raw.resolve().glob('**/*.nii.gz') if f.is_file()]
training_v20_raw = [f for f in training_v20_raw.resolve().glob('**/*.nii.gz') if f.is_file()]

training_raw = training_v19_raw + training_v20_raw

training_v19_points = Path('VerSe19/dataset-verse19training/derivatives/')
training_v20_points = Path('VerSe20/dataset-01training/derivatives/')

training_v19_points = [f for f in training_v19_points.resolve().glob('**/*.json') if f.is_file()]
training_v20_points = [f for f in training_v20_points.resolve().glob('**/*.json') if f.is_file()]

training_points = training_v19_points + training_v20_points

save_heatmap_data_training(training_raw, training_points)

print('Preprocessing validation data')

valid_v19_raw = Path('VerSe19/dataset-verse19validation/rawdata/')
valid_v20_raw = Path('VerSe20/dataset-02validation/rawdata/')


valid_v19_raw = [f for f in valid_v19_raw.resolve().glob('**/*.nii.gz') if f.is_file()]
valid_v20_raw = [f for f in valid_v20_raw.resolve().glob('**/*.nii.gz') if f.is_file()]

valid_raw = valid_v19_raw + valid_v20_raw

valid_v19_points = Path('VerSe19/dataset-verse19validation/derivatives/')
valid_v20_points = Path('VerSe20/dataset-02validation/derivatives/')

valid_v19_points = [f for f in valid_v19_points.resolve().glob('**/*.json') if f.is_file()]
valid_v20_points = [f for f in valid_v20_points.resolve().glob('**/*.json') if f.is_file()]

valid_points = valid_v19_points + valid_v20_points

save_heatmap_data_validation(valid_raw, valid_points)
