"""gets mwaspkl file from s3 and converts it to a readable csv file"""
import os
import sys
import pickle
import subprocess

S3_BUCKET = 's3://serratus-biosamples/condensed-bioproject-metadata/'


def mwaspkl_to_readable_csv(bioproject, output_dir):
    """converts mwaspkl file to readable csv file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # check if size is 1
    if os.path.getsize(f'{bioproject}.mwaspkl') == 1:
        print(f"{bioproject}.mwaspkl was empty. Exiting...")
        return
    # get the biosamples_ref and set_df from the mwaspkl file
    with open(f'{bioproject}.mwaspkl', 'rb') as f:
        biosamples_ref = pickle.load(f)
        set_df = pickle.load(f)

    set_df['biosample_index_list'] = set_df.apply(
        lambda row:
        [biosamples_ref[i] for i in row['biosample_index_list']]
        if row['include?'] else
        [sample for i, sample in enumerate(biosamples_ref) if i not in row['biosample_index_list']],
        axis=1)
    # remove include? column
    set_df = set_df.drop(columns=['include?'])

    # save the new dataframe to a csv file
    set_df.to_csv(f"{output_dir}/{bioproject}.csv", index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        local_mode = False
        if '-s' in sys.argv:
            local_mode = True
        sys.argv.remove('-s')

        if not local_mode:
            # get the mwaspkl file from s3
            bioproject = sys.argv[1]
            try:
                subprocess.run('aws s3 cp ' + f'{S3_BUCKET}{bioproject}.mwaspkl .', shell=True)
            except Exception as e:
                print(f"Error getting the mwaspkl file: {e}")
                sys.exit(1)
        else:
            bioproject = sys.argv[1]
        if len(sys.argv) < 3:
            output_dir = 'readable_condensed_bioproject_metadata_csvs'
        else:
            output_dir = sys.argv[2]

        mwaspkl_to_readable_csv(sys.argv[1], output_dir)

        if not local_mode:
            os.remove(sys.argv[1] + '.mwaspkl')
        print(f"Converted {bioproject} to a readable csv file.")
    else:
        print("Please provide a file to read.")
