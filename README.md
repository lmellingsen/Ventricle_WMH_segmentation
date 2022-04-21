This code was used for generating the results of the article:
# A joint ventricle and WMH segmentation from MRI for evaluation of healthy and pathological changes in the aging brain
which is currently under review at PLOS ONE.
The SegAE part of the pipeline was first described in:
https://www.sciencedirect.com/science/article/pii/S2213158219304322#keys0001

## Installation
The environment was installed using anaconda.
Required imports can be seen at the top of each script, with further details in requirements.txt.

## Usage
The user must replace the filepaths in the scripts to their own specified paths to T1w, T2w and FLAIR brain MRIs.
Furthermore, the the user must specify a path for the segmentations to be saved in.
