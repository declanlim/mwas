# Metadata Wide Association Study (MWAS)
Instructions for sending requests to the MWAS server:
1) Make sure your data file is a csv that contains 3 columns in this order: run, group, quantifier
2) convert your csv into a json file so it can be attached to the request (as the request's **data**)
3) The Uri for the request should be: "http://ec2-75-101-194-81.compute-1.amazonaws.com:5000/run_mwas"
4) Send the request (POST request).
On Windows, you can follow these steps in the Powershell

   $jsonData = Import-Csv -Path {path to your csv} | ConvertTo-Json <br />
   $url = "http://ec2-75-101-194-81.compute-1.amazonaws.com:5000/run_mwas" <br />
   $response = Invoke-WebRequest -Uri $url -Method Post -Body $jsonData -ContentType "application/json" -UseBasicParsing <br />

Then, mwas should start running. As of now, the user won't be able to know the actual MWAS output or the progress while it's being processed, but you can check $response to check if it finished/how it exited (if a problem did occur).

 <br />
  <br />
(THE REST OF THIS README NEEDS TO BE UPDATED)

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
