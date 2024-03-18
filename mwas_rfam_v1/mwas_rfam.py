# pip install --upgrade psycopg2-binary pandas numpy scipy requests boto3 s3fs fsspec

# import required libraries
import psycopg2
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
import warnings
import logging
import sys
import io
import pickle
import csv
import boto3

# --------------- HELPER FUNCTIONS ---------------
def family_biosamples_mapper(row):
    if row['bio_sample'] in family_biosamples_n_reads:
        family_biosamples_n_reads[row['bio_sample']].append(row['n_reads'])
        family_biosample_spots[row['bio_sample']].append(row['spots'])
        # family_biosample_runs[row['bio_sample']].append(row['run_id']) # TO BE USED IF MAPPING IS DONE FROM RUNS
    else:
        family_biosamples_n_reads[row['bio_sample']] = [row['n_reads']]
        family_biosample_spots[row['bio_sample']] = [row['spots']]
        # family_biosample_runs[row['bio_sample']] = [row['run_id']] # TO BE USED IF MAPPING IS DONE FROM RUNS

    
def family_bioproject_mapper(row):
    try:
        return srarun_bioprojects_map[row['bio_sample']]
    except:
        return np.nan

def get_n_reads(row):
    # try to get the n_reads from family_biosamples_n_reads, then from srarun_biosamples_map, then default to np.nan
    #   - tries to get exact n_reads from serratus
    #   - if run in serratus but no n_reads, then set to 0 (from srarun_biosamples_map)
    #   - if not run in serratus, then set to np.nan
    try:
        return family_biosamples_n_reads[row['biosample_id']]
    except:
        try:
            return srarun_biosamples_map[row['biosample_id']]
        except:
            return np.nan

def get_n_spots(row):
    # try to get the n_spots from family_biosample_spots, otherwise default to 0 
    ##### CHECK THIS ##### will have divide by 0 error if the mapping is wrong
    try:
        return family_biosample_spots[row['biosample_id']]
    except:
        return 0

def get_log_fold_change(true, false):
    # calculate the log fold change of true with respect to false
    #   - if true and false is 0, then return 0

    if true == 0 and false == 0:
        return 0
    elif true == 0:
        return -np.inf
    elif false == 0:
        return np.inf
    else:
        return np.log2(true / false)

# ---------------------------------------------

warnings.filterwarnings('ignore') # FIX THIS LATERE TO NOT IGNORE WARNINGS

# database connection details
host = "serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com"
database = "summary"
user = "public_reader"
password = "serratus"

# read in the file name from the command line
if len(sys.argv) != 2:
    print('Please provide a file name')
    sys.exit(1)
family_file = sys.argv[1]

# file name provided with ls family_groups/ | parallel -j 3 "python mwas_rfam.py"

# create a logger with the file name
logger = logging.getLogger(family_file)
if logger.hasHandlers():
    logger.handlers.clear()
fh = logging.FileHandler('logs/' + family_file.replace('.txt', '.log'))
formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# get the file number from the file name
file_num = family_file.replace('.txt', '').split('_')[-1]

# # load information from pickles
with open(f'../pickles/rfam_df_{file_num}.pickle', 'rb') as f:
    rfam_df = pickle.load(f)
with open(f'../pickles/srarun_df_{file_num}.pickle', 'rb') as f:
    srarun_df = pickle.load(f)

with open(f'../pickles/srarun_biosamples_map_{file_num}.pickle', 'rb') as handle:
    # mapping from biosamples (in srarun_df) to 0 for default count
    srarun_biosamples_map = pickle.load(handle) 
with open(f'../pickles/srarun_bioprojects_map_{file_num}.pickle', 'rb') as handle:
    # mapping from biosamples (in srarun_df) to bioprojects
    srarun_bioprojects_map = pickle.load(handle)

logger.info(f'Loaded pickles')

# set up variables for API call
base_url = 'https://serratus-biosamples.s3.us-east-1.amazonaws.com/bioprojects_csv/'
headers = {'Host': 'serratus-biosamples.s3.amazonaws.com'}

s3 = boto3.resource('s3')
bucket = s3.Bucket('serratus-biosamples')
logger.info(f'Connected to S3 bucket')

# files are located in the family_groups directory
with open('family_groups/' + family_file, 'r') as f:
    print(f'Processing {family_file}')
    logger.info(f'Processing {family_file}')
    for family in f:
        # remove the new line character
        family = family.strip()
        logger.info(f'Processing {family}')

        # subset the rfam_df to the family of interest
        family_df = rfam_df[rfam_df['family_group'] == family]
        family_df = pd.merge(family_df[['run_id', 'n_reads']], srarun_df[['run', 'bio_sample', 'spots']], left_on='run_id', right_on='run')

        # map the biosamples in the family to n_reads and spots
        family_biosamples_n_reads = {}
        family_biosample_spots = {}
        family_df.apply(family_biosamples_mapper, axis=1)
        # average the n_reads and spots for each biosample
        for biosample in family_biosamples_n_reads:
            family_biosamples_n_reads[biosample] = np.mean(family_biosamples_n_reads[biosample])
            family_biosample_spots[biosample] = np.mean(family_biosample_spots[biosample])

        # map the biosamples in the family to their bioprojects
        family_df['bio_project'] = family_df.apply(family_bioproject_mapper, axis=1)

        # get the bioprojects for the family and remove None from the list
        bioprojects = list(family_df['bio_project'].unique())
        if None in bioprojects:
            bioprojects.remove(None)

        # iterate through the bioprojects for the family
        for bioproject_id in bioprojects:
            # --------------- SET UP DF FOR MWAS ---------------
            # get the csv file for the bioproject
            url = base_url + bioproject_id + '.csv'
            
            # connection reset by peer error?
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    # skip the bioproject if it doesn't exist
                    logger.warning(f'Error: {response.status_code} for {bioproject_id}')
                    continue
            except Exception as e:
                logger.warning(f'Error occurred while making API call: {e}')
                continue


            # read the csv file into a dataframe
            bp_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), quoting=csv.QUOTE_NONE)

            # skip if there are two rows or less
            num_rows = bp_df.shape[0]
            if num_rows <= 2:
                logger.warning(f'Skipping {bioproject_id}: {num_rows} rows')
                continue
        
            # add the n_reads and n_spots columns to the dataframe
            bp_df['n_reads'] = bp_df.apply(get_n_reads, axis=1)
            bp_df['n_spots'] = bp_df.apply(get_n_spots, axis=1)

            # add the rpm column to the dataframe
            bp_df['rpm'] = bp_df.apply(lambda row: row['n_reads'] / row['n_spots'] * 1000000 if row['n_spots'] != 0 else 0, axis=1)
            
            # --------------- SELECTING COLUMNS TO TEST ---------------
            # get the list of testable columns in the dataframe
            remove_cols = {'biosample_id','n_reads','n_spots','rpm'}
            target_cols = list(set(bp_df.columns) - remove_cols)

            metadata_counts = {}

            # store the value counts if the column can be tested
            for col in target_cols:
                counts = bp_df[col].value_counts()
                n = len(counts)
                # skip if there is only one unique value or if all values are unique
                if n == 1:
                    continue
                if n == num_rows:
                    continue
                metadata_counts[col] = counts

            existing_samples = list()
            target_col_runs = {}

            # iterate through the values for all columns that can be tested
            for target_col, value_counts in metadata_counts.items():
                for target_term, count in value_counts.items():
                    # skip if there is only one value
                    if count == 1:
                        continue

                    # get the biosamples that correspond to the target term
                    target_term_biosamples = list(bp_df[bp_df[target_col] == target_term]['biosample_id'])

                    # check if the same biosamples are already stored and aggregate the columns
                    target_col_name = f'{target_col}\t{target_term}'

                    if target_term_biosamples in existing_samples:
                        existing_key = list(target_col_runs.keys())[list(target_col_runs.values()).index(target_term_biosamples)]
                        target_col_name = f'{existing_key}\r{target_col_name}'
                        # update the dictionary with the new name
                        target_col_runs[target_col_name] = target_col_runs.pop(existing_key)
                    else:
                        existing_samples.append(target_term_biosamples)
                        target_col_runs[target_col_name] = target_term_biosamples

            # --------------- RUN TTESTS ---------------
            # set up the columns for the output df
            output_cols = ['bioproject_id', 'family', 'metadata_field', 'metadata_value', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false', 'sd_rpm_true', 'sd_rpm_false', 'fold_change', 't_statistic', 'p_value']
            output_dict = {col: [] for col in output_cols}

            for target_col, biosamples in target_col_runs.items():
                num_true = len(biosamples)
                num_false = num_rows - num_true
                # get the rpm values for the target and remainng biosamples
                true_rpm = bp_df[bp_df['biosample_id'].isin(biosamples)]['rpm']
                false_rpm = bp_df[~bp_df['biosample_id'].isin(biosamples)]['rpm']
                
                # calculate desecriptive stats
                # NON CORRECTED VALUES
                mean_rpm_true = np.nanmean(true_rpm)
                mean_rpm_false = np.nanmean(false_rpm)
                sd_rpm_true = np.nanstd(true_rpm)
                sd_rpm_false = np.nanstd(false_rpm)

                # calculate fold change and check if any values are nan
                fold_change = get_log_fold_change(mean_rpm_true, mean_rpm_false)
                # calculate t-statistic and p-value
                t_statistic, p_value = stats.ttest_ind_from_stats(mean1=mean_rpm_true, std1=sd_rpm_true, nobs1=num_true, mean2=mean_rpm_false, std2=sd_rpm_false, nobs2=num_false, equal_var=False)

                # extract metadata_field (column names) and metadata_value (column values)
                metadata_tmp = target_col.split('\r') # pairs of metadata_field and metadata_value for aggregated columns
                # metadata_field = '\t'.join(metadata_tmp[::2])
                # metadata_value = '\t'.join(metadata_tmp[1::2])
                metadata_field = '\t'.join(pair.split('\t')[0] for pair in metadata_tmp)
                metadata_value = '\t'.join(pair.split('\t')[1] for pair in metadata_tmp)


                # add values to output dict
                output_dict['bioproject_id'].append(bioproject_id)
                output_dict['family'].append(family)
                output_dict['metadata_field'].append(metadata_field)
                output_dict['metadata_value'].append(metadata_value)
                output_dict['num_true'].append(num_true)
                output_dict['num_false'].append(num_false)
                output_dict['mean_rpm_true'].append(mean_rpm_true)
                output_dict['mean_rpm_false'].append(mean_rpm_false)
                output_dict['sd_rpm_true'].append(sd_rpm_true)
                output_dict['sd_rpm_false'].append(sd_rpm_false)
                output_dict['fold_change'].append(fold_change)
                output_dict['t_statistic'].append(t_statistic)
                output_dict['p_value'].append(p_value)

            # create the output df and sort by p_value
            mwas_df = pd.DataFrame(output_dict)
            mwas_df = mwas_df.sort_values(by='p_value')

            # log if there are significant results
            num_significant = mwas_df[mwas_df['p_value'] < 0.05].shape[0]
            if num_significant > 0:
                logger.info(f'{bioproject_id} - {num_significant} significant results')
            
            # write the mwas_df to a csv file on S3
            key = f's3://serratus-biosamples/mwas/{family}/{bioproject_id}.csv'
            mwas_df.to_csv(key, index=False)
            logger.info(f'Finished {bioproject_id}')
        logger.info(f'Finished MWAS for {family}')
        # break

print(f'Finished processing {family_file}')
logger.info(f'Finished processing {family_file}')