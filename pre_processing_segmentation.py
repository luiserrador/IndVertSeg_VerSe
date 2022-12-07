from pathlib import Path
from utils.pre_processing_segmentation_utils import *


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

save_iso_croped_data(training_raw, training_derivatives, arrayData)
