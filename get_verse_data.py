import wget
import sys
import os
import zipfile


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


if not os.path.exists('VerSe19'):
    os.mkdir('Verse19')

    print('Downloading VerSe19 Training Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip', out='VerSe19/Training',
                  bar=bar_progress)

    print('\nDownloading VerSe19 Validation Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip',
                  out='VerSe19/Validation',
                  bar=bar_progress)

    print('\nDownloading VerSe19 Test Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip', out='VerSe19/Test',
                  bar=bar_progress)

if not os.path.exists('VerSe20'):
    os.mkdir('Verse20')

    print('\nDownloading VerSe20 Training Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip', out='VerSe20/Training',
                  bar=bar_progress)

    print('\nDownloading VerSe20 Validation Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip',
                  out='VerSe20/Validation',
                  bar=bar_progress)

    print('\nDownloading VerSe20 Test Data')
    wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip', out='VerSe20/Test',
                  bar=bar_progress)

print('\nExtracting VerSe19 data')

archive = zipfile.ZipFile('VerSe19/Training')

for file in archive.namelist():
    if file.startswith('dataset-verse19training/derivatives/'):
        archive.extract(file, 'VerSe19/')
    elif file.startswith('dataset-verse19training/rawdata/'):
        archive.extract(file, 'VerSe19/')

archive = zipfile.ZipFile('VerSe19/Validation')

for file in archive.namelist():
    if file.startswith('dataset-verse19validation/derivatives/'):
        archive.extract(file, 'VerSe19/')
    elif file.startswith('dataset-verse19validation/rawdata/'):
        archive.extract(file, 'VerSe19/')

archive = zipfile.ZipFile('VerSe19/Test')

for file in archive.namelist():
    if file.startswith('dataset-verse19test/derivatives/'):
        archive.extract(file, 'VerSe19/')
    elif file.startswith('dataset-verse19test/rawdata/'):
        archive.extract(file, 'VerSe19/')

print('\nExtracting VerSe20 data')

archive = zipfile.ZipFile('VerSe20/Training')

for file in archive.namelist():
    if (file.startswith('dataset-01training/derivatives/') or file.startswith('dataset-01training/rawdata/')) and (file.endswith('.nii.gz') or file.endswith('.json')):
        archive.extract(file, 'VerSe20/')

archive = zipfile.ZipFile('VerSe20/Validation')

for file in archive.namelist():
    if (file.startswith('dataset-02validation/derivatives/') or file.startswith('dataset-02validation/rawdata/')) and (file.endswith('.nii.gz') or file.endswith('.json')):
        archive.extract(file, 'VerSe20/')

archive = zipfile.ZipFile('VerSe20/Test')

for file in archive.namelist():
    if (file.startswith('dataset-03test/derivatives/') or file.startswith('dataset-03test/rawdata/')) and (file.endswith('.nii.gz') or file.endswith('.json')):
        archive.extract(file, 'VerSe20/')



