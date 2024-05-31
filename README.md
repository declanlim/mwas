# Metadata Wide Association Study (MWAS)
(THIS README NEEDS TO BE UPDATED)

Metadata wide association studies are statistical studies that find correlations between a given observation and metadata from the NCBI BioSample database (ADD LINK). Using extracted information from the BioSample Database, t-tests or permutation tests are used to calculate the statistical significance of any correlations in the data. 

This repository was used for running MWAS on the Serratus rfamily database (ADD LINK, DATE)

## Required files:
- S3 LINK FOR BUCKET
    - pickle files containing extracted metadata from BioSample (as BioProjects)
    - pickle files containing calculated viral counts from Serratus
- This repository
    - ```mwas_rfam.py``` file for running the actual analysis
    - ```family_groups``` folder containing families (groups) of viruses


## Generalizing ```mwas_rfam.py```
- The MWAS analysis is largely split into 3 parts:
    - preprocessing
    - statistical analysis
    - postprocessing

### Preprocessing
The preprocessing for MWAS can mostly be shared between the runs. The extraction of data from the BioSample database (and grouping into BioProjects) can technically be repeated daily, but is a computationally heavy piece of code. There are a small number of errors in BioSample parsing (where BioSamples have " and , in the data), but these can easily be fixed on the next run of BioProject csv generation.

More specific to MWAS, there needs to be a way of linking quantifications of an observation to the BioSamples. Currently this exists in the form of a dictionary that is stored as a pickle file (ADD LINK). THis is used in line (ADD LINE) in ```mwas_rfam.py``` where the viral counts for a specific family are attached to the rfamily (NOT RIGHT?) dataframe. These will need to be substituted for the provided counts by the end user. 

The ```mwas_rfam.py``` file is also designed to work with a given family of viruses. This can be used efficiently as the large rfamily database can quickly be subset to only the rows of a certain family. For the generalized MWAS run, this should be even simpler, as end users will provide the specific runs and grouping that they wish to run the analysis on. 

### Statistical Analysis
All the statistical analysis for MWAS should remain the same in the generalized version.

### Postprocessing
All the processing for MWAS should remain the same in the generalized version.
