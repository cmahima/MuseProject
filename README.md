# MuseProject
This repository consists of codes that can be used to preprocess data from Muse headband and apply ML models to it to classify data on basis of colours RGB

# editmusefilewithtime.py
This file is used to edit the RAW file from muse application MIND MONITER.The file is divided into sub files that contain data at instances when person looks at red, green and blue colour. Since one experiment consists of 20 trials for each colour in our case, therefore we get 60 csv files with 20 files of each of red, green and blue respectively

# musecombinedimage.m
In order to obtain spectrogram images from the data we use MATLAB. The images are obtained for each electrode as well as combination of electrodes by applying Morlet Wavelet Transform.

# museexpfinal_lastrun.py
This file is used to run the visual experiment. It used Psychopy library of Python.

# runmuseapp.sh
This shell script runs the codes to preprocess the data and get the data in trainable format

# mlmodelmuse
This folder has models that have been applied on the data

## Note
This repository will be updated as there is progress in work
