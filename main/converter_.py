import os
import sys
import time
import pandas as pd
import pickle
from metadata_set_maker import metadata_to_set_accession

BLACKLIST = {"PRJEB37886", "PRJNA514245", "PRJNA716984", "PRJNA731148", "PRJNA631508", "PRJNA665224", "PRJNA716985", "PRJNA479871", "PRJNA715749", "PRJEB11419",
             "PRJNA750736", "PRJNA525951", "PRJNA720050", "PRJNA731152", "PRJNA230403", "PRJNA675921", "PRJNA608064", "PRJNA486548",
             "nan", "PRJEB43828", "PRJNA609094", "PRJNA686984", "PRJNA647773", "PRJNA995950"}
# 23 bioprojects that are over 100MB in size (original csv)


def process_file(_metadata_file, source, _storage) -> tuple[int, int, float, str, int, int, int, set]:
    """processing file (load csv to df, convert to set-form (condensed for MWAS), """
    # set up
    original_file = f"{source + '/' if source else ''}{_metadata_file}"
    new_file = f"{_storage}/{_metadata_file.split('/')[-1][:-4]}.mwaspkl"
    _size = os.path.getsize(f'{source}/{_metadata_file}')
    start_time = time.time()
    if _size == 1:
        with open(new_file, 'wb') as f:
            f.write(b'0')
        return _size, 1, time.time() - start_time, "Original csv was empty.", 0, 0, 0, set()
    elif bioproject in BLACKLIST:
        with open(new_file, 'wb') as f:
            f.write(b'1')
        return _size, 1, time.time() - start_time, "Bioproject is in the blacklist.", 0, 0, 0, set()
    try:
        metadata_dataframe = pd.read_csv(original_file, low_memory=False)
    except Exception as e:
        return os.path.getsize(original_file), 0, time.time() - start_time, f"FAILED - csv reading: {e}", 0, 0, 0, set()

    # check if metadata_dataframe is empty
    if metadata_dataframe.shape[0] < 4:  # because it's impossible to do a meaningful test with less than 4 entities
        with open(new_file, 'wb') as f:
            f.write(b'0')
        return _size, 1, time.time() - start_time, "Less than 4 rows in csv file => empty file.", metadata_dataframe.shape[0], 0, 0, set()

    # convert metadata to condensed set form
    try:
        biosamples_ref, set_df, _comment, _, is_empty = metadata_to_set_accession(metadata_dataframe)
    except Exception as e:
        print(f"Error processing {_metadata_file}: {e}")
        return _size, 0, time.time() - start_time, f"FAILED - condensing: {e}", metadata_dataframe.shape[0], 0, 0, set()

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

    if _size == 8917 and _pickle_size == 2241 and len(biosamples_ref) == 22:
        _comment += "Very likely to be a dupe-bug file."
    return _size, _pickle_size, time.time() - start_time, _comment, len(biosamples_ref), set_df.shape[0], set_df[set_df['test_type'] == 'permutation-test'].shape[0], set(set_df['attributes'])


if __name__ == '__main__':
    if len(sys.argv) > 2:
        arg1 = sys.argv[1]
        storage = sys.argv[2]
        print(f"{arg1}")
        write_mode = 'a' if '--start_at' in sys.argv else 'w'
        with open('conversion_results.csv', write_mode) as results_f, open('conversion_errors.txt', 'w') as errors_f:
            results_f.write('file,original_size,condensed_pickle_size,processing_time,comment,num_biosamples,num_sets,num_permutation_sets\n')

            # get files to iterate over
            files = []
            if os.path.isdir(arg1):  # convert all csv files in the directory
                src = arg1
                files = os.listdir(arg1)
                print("sorting files")
                files.sort()
                print("finished sorting files")
                if '--start_at' in sys.argv and sys.argv.index('--start_at') + 1 < len(sys.argv):
                    start_at = sys.argv[sys.argv.index('--start_at') + 1]
                    try:
                        print(f"Resuming progress at {start_at}")
                        files = files[files.index(start_at):]
                    except ValueError:
                        try:
                            print(f"Resuming progress at {start_at}.csv")
                            files = files[files.index(f"{start_at}.csv"):]
                        except ValueError:
                            print(f"Could not find {start_at} in the list of files. Exiting...")
                            sys.exit(1)
            elif arg1.endswith('.txt'):
                src = ''
                with open(arg1, 'r') as f:
                    files = f.readlines()
                    print(f"Found {len(files)} files to process.")
            elif arg1.endswith('.csv'):
                src = ''
                files = [arg1]

            # process files
            if not files:
                print("No files found to process.")

            fields = set()

            for file in files:
                file = file.replace('\n', '')
                bioproject = file.split('/')[-1][:-4]
                print(f"Processing {bioproject}...")
                if file.endswith('.csv'):
                    try:
                        size, pickle_size, conversion_time, comment, num_biosamples, num_sets, num_permutation_sets, field_names = process_file(f"{file}", src, storage)
                        if comment == '':
                            comment = 'No issues.'
                        results_f.write(f"{bioproject},{size},{pickle_size},{conversion_time},{comment},{num_biosamples},{num_sets},{num_permutation_sets}\n")
                        print(f"Processed {bioproject} in {conversion_time} seconds.")
                        fields.update(field_names)
                    except Exception as e:
                        print(f"Failed to process {bioproject} due to error: {e}")
                        results_f.write(f"{bioproject},{os.path.getsize(f'{arg1}/{file}')},0,0,FAILED - misc error: {e}\n,0,0,0\n")
                        errors_f.write(f"{bioproject},{e}\n")

            # record fields to file in form {field1, field2, field3}
            with open('fields.txt', 'w') as fields_f:
                fields_f.write(str(fields))

            # split every element in fields by ';' to get actual fields
            actual_fields = set()
            for field in fields:
                actual_fields.update(str(field).split(';'))
            # strip whitespace
            actual_fields = {field.strip() for field in actual_fields}
            with open('actual_fields.txt', 'w') as actual_fields_f:
                for field in actual_fields:
                    actual_fields_f.write(f"{field}\n")
    print("Conversion complete.")
