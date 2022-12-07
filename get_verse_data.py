import wget
import sys
import os

#create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


if not os.path.exists('VerSe19'):
  os.mkdir('Verse19')

if not os.path.exists('VerSe20'):
  os.mkdir('Verse20')

print('Downloading VerSe19 Training Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip', out='VerSe19/Training', bar=bar_progress)

print('Downloading VerSe19 Validation Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip', out='VerSe19/Validation', bar=bar_progress)

print('Downloading VerSe19 Test Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip', out='VerSe19/Test', bar=bar_progress)

print('Downloading VerSe20 Training Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip', out='VerSe20/Training', bar=bar_progress)

print('Downloading VerSe20 Validation Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip', out='VerSe20/Validation', bar=bar_progress)

print('Downloading VerSe20 Test Data')
wget.download('https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip', out='VerSe20/Test', bar=bar_progress)

