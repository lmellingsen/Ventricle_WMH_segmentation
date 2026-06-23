This code was used for generating the results of the article:
### A joint ventricle and WMH segmentation from MRI for evaluation of healthy and pathological changes in the aging brain
which is published in PLOS ONE: Atlason HE, Love A, Robertsson V, Blitz AM, Sigurdsson S, Gudnason V, Ellingsen LM, (2022) A joint ventricle and WMH segmentation from MRI for evaluation of healthy and pathological changes in the aging brain. PLOS ONE 17(9): e0274212. https://doi.org/10.1371/journal.pone.0274212.
The SegAE part of the pipeline was first described in: 
Atlason et al., SegAE: Unsupervised white matter lesion segmentation from brain MRIs using a CNN autoencoder,
NeuroImage: Clinical, Volume 24, 2019, 102085, ISSN 2213-1582, https://doi.org/10.1016/j.nicl.2019.102085.

## Installation
The environment was installed using anaconda.
Required imports can be seen at the top of each script, with further details in requirements.txt.

## Usage
The user must replace the filepaths in the scripts to their own specified paths to T1w, T2w and FLAIR brain MRIs.
Furthermore, the user must specify a path for the segmentations to be saved in.
