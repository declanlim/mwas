import sys
import pickle

if len(sys.argv) < 3:
    print("Usage: python unload_pickles.py <file_list> <csv folder>")
    sys.exit(1)
pickle_files = sys.argv[1]  # Get the file name from the command-line arguments

with open(pickle_files, 'r') as file_list, open('csv_files.txt', 'w') as csv_files:
    # read line by line (each line is a pickle file address)
    for line in file_list:
        info = line.split(' ')
        pickle_file = info[0]
        pickle_size = info[1] if len(info) > 1 else None
        with open(pickle_file, 'rb') as file:
            try:
                bioproj_df = pickle.load(file)
            except Exception as e:
                print(f"Failed to load {pickle_file}: {e}")
                exit(1)
            bioproject_name = str(pickle_file).split('/')[-1].split('.')[0]
            bioproj_df.to_csv(f'{sys.argv[2]}/{bioproject_name}.csv', index=False)
            csv_files.write(f'{sys.argv[2]}/{bioproject_name}.csv{" " + str(pickle_size) if pickle_size is not None else ""}\n')
print("loaded bioproject csvs")
