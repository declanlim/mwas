import os
import sys
import time
import pandas as pd
import pickle
from metadata_set_maker import metadata_to_set_accession

# TODO: debug for this:
# PRJDA61421,6090,0,0,failed


def process_file(_metadata_file, source, _storage):
    """processing file (load csv to df, convert to set-form (condensed for MWAS), """
    # set up
    original_file = f"{source}/{_metadata_file}"
    new_file = f"{_storage}/{_metadata_file.split('/')[-1][:-4]}.mwaspkl"
    _size = os.path.getsize(f'{source}/{_metadata_file}')
    start_time = time.time()
    try:
        metadata_dataframe = pd.read_csv(original_file, low_memory=False)
    except Exception as e:
        return os.path.getsize(original_file), 0, time.time() - start_time, f"FAILED - csv reading: {e}"

    # check if metadata_dataframe is empty
    if metadata_dataframe.shape[0] <= 2:
        with open(new_file, 'wb') as f:
            f.write(b'0')
        print(f"Less than 3 rows in csv file: {_metadata_file}")
        return _size, 1, time.time() - start_time, "Less than 3 rows in csv file => empty file."

    # convert metadata to condensed set form
    try:
        biosamples_ref, set_df, _comment, _, is_empty = metadata_to_set_accession(metadata_dataframe)
    except Exception as e:
        print(f"Error processing {_metadata_file}: {e}")
        return _size, 0, time.time() - start_time, f"FAILED - condensing: {e}"

    # pickle the biosamples_ref and set_df into 1 file. Don't worry about other file data for now
    # create the new file in dir called storage
    with open(new_file, 'wb') as f:
        # we want to dump both objects into the same file but so that we can extract them separately later
        if is_empty:
            f.write(b'0')
        else:
            pickle.dump(biosamples_ref, f)
            pickle.dump(set_df, f)

    _pickle_size = 1 if is_empty else os.path.getsize(new_file)

    if _size == 8917 and _pickle_size == 2122:
        _comment += "- Very likely to be a dupe-bug file."
    return _size, _pickle_size,  time.time() - start_time, _comment


if __name__ == '__main__':
    if len(sys.argv) > 2:
        arg1 = sys.argv[1]
        storage = sys.argv[2]
        print(f"{arg1}")
        # check if arg1 is a dir or a file
        with open('conversion_results.csv', 'w') as results_f, open('conversion_errors.txt', 'w') as errors_f:
            results_f.write('file,original_size,condensed_pickle_size,processing_time,comment\n')

            # get files to iterate over
            files = []
            if os.path.isdir(arg1):  # convert all csv files in the directory
                files = os.listdir(arg1)
            elif arg1.endswith('.txt'):
                with open(arg1, 'r') as f:
                    files = f.readlines()
            elif arg1.endswith('.csv'):
                files = [arg1]

            # process files
            if not files:
                print("No files found to process.")
            for file in files:
                file = file.replace('\n', '')
                bioproject = file.split('/')[-1][:-4]
                print(f"Processing {bioproject}...")
                if file.endswith('.csv'):
                    try:
                        size, pickle_size, conversion_time, comment = process_file(f"{file}", arg1, storage)
                        results_f.write(f"{bioproject},{size},{pickle_size},{conversion_time},{comment}\n")
                        print(f"Processed {bioproject} in {conversion_time} seconds.")
                    except Exception as e:
                        print(f"Failed to process {bioproject} due to error: {e}")
                        results_f.write(f"{bioproject},{os.path.getsize(f'{arg1}/{file}')},0,0,FAILED - misc error: {e}\n")
                        errors_f.write(f"{bioproject},{e}\n")
    print("Conversion complete.")
