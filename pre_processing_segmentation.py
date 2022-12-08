from pathlib import Path
from utils.pre_processing_segmentation_utils import *


print('Preprocessing training data')

training_v19_raw = Path('VerSe19/dataset-verse19training/rawdata/')
training_v20_raw = Path('VerSe20/dataset-01training/rawdata/')

training_v19_raw = [f for f in training_v19_raw.resolve().glob('**/*.nii.gz') if f.is_file()]
training_v20_raw = [f for f in training_v20_raw.resolve().glob('**/*.nii.gz') if f.is_file()]

training_raw = training_v19_raw + training_v20_raw

training_v19_derivatives = Path('VerSe19/dataset-verse19training/derivatives/')
training_v20_derivatives = Path('VerSe20/dataset-01training/derivatives/')

training_v19_derivatives = [f for f in training_v19_derivatives.resolve().glob('**/*.nii.gz') if f.is_file()]
training_v20_derivatives = [f for f in training_v20_derivatives.resolve().glob('**/*.nii.gz') if f.is_file()]

training_derivatives = training_v19_derivatives + training_v20_derivatives

sorted(training_raw)
sorted(training_derivatives)

mul_vert = get_to_balance_array(training_derivatives)

arrayData = get_array_data_training(training_derivatives, mul_vert)

save_iso_croped_data_training(training_raw, training_derivatives, arrayData)

print('Preprocessing validation data')

valid_v19_raw = Path('VerSe19/dataset-verse19validation/rawdata/')
valid_v20_raw = Path('VerSe20/dataset-02validation/rawdata/')

valid_v19_raw = [f for f in valid_v19_raw.resolve().glob('**/*.nii.gz') if f.is_file()]
valid_v20_raw = [f for f in valid_v20_raw.resolve().glob('**/*.nii.gz') if f.is_file()]

valid_raw = valid_v19_raw + valid_v20_raw

valid_v19_derivatives = Path('VerSe19/dataset-verse19validation/derivatives/')
valid_v20_derivatives = Path('VerSe20/dataset-02validation/derivatives/')

valid_v19_derivatives = [f for f in valid_v19_derivatives.resolve().glob('**/*.nii.gz') if f.is_file()]
valid_v20_derivatives = [f for f in valid_v20_derivatives.resolve().glob('**/*.nii.gz') if f.is_file()]

valid_derivatives = valid_v19_derivatives + valid_v20_derivatives

sorted(valid_raw)
sorted(valid_derivatives)

arrayDataValid = get_array_data_valid(valid_derivatives)

save_iso_croped_data_validation(valid_raw, valid_derivatives, arrayDataValid)
