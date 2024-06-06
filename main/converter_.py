import os
import sys
import time
import pandas as pd
import pickle
from main.metadata_set_maker import metadata_to_set_accession


def process_file(_metadata_file, _storage):
    """processing file (load csv to df, convert to set-form (condensed for MWAS), """
    start_time = time.time()
    metadata_dataframe = pd.read_csv(_metadata_file)
    # get size of metadata file
    _size = os.path.getsize(_metadata_file)
    biosamples_ref, set_df, _comment = metadata_to_set_accession(metadata_dataframe)

    # pickle the biosamples_ref and set_df into 1 file. Don't worry about other file data for now
    # create the new file in dir called storage
    with open(f"{_storage}/{_metadata_file.split('/')[-1]}.mwas", 'wb') as f:
        # we want to dump both objects into the same file but so that we can extract them separately later
        pickle.dump(biosamples_ref, f)
        pickle.dump(set_df, f)
        # get pickle size
    _pickle_size = os.path.getsize(f"{_storage}/{_metadata_file.split('/')[-1]}.mwas")
    _conversion_time = time.time() - start_time
    return _size, _pickle_size, _conversion_time, _comment


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        # check if arg1 is a dir or a file
        with open('conversion_results.csv', 'w') as results_f:
            results_f.write('file,pickle_size,output_size,processing_time,comment\n')
            if os.path.isdir(arg1):  # convert all csv files in the directory
                files = os.listdir(arg1)
            elif arg1.endswith('.txt'):
                with open(arg1, 'r') as f:
                    files = f.readlines()
            elif arg1.endswith('.csv'):
                files = [arg1]

            for file in files:
                print(f"Processing {file}...")
                if file.endswith('.csv'):
                    try:
                        size, pickle_size, conversion_time, comment = process_file(f"{file}", arg1)
                        results_f.write(f"{file},{size},{pickle_size},{conversion_time},{comment}\n")
                        print(f"Processed {file} in {conversion_time} seconds.")
                    except Exception as e:
                        print(f"Failed to process {file} due to error: {e}")
                        results_f.write(f"{file},{os.path.getsize(file)},0,0,failed\n")
    print("Conversion complete.")
