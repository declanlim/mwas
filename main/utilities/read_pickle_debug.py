"""reads the pickle file and prints it"""
import pickle
import sys
import pandas as pd


def read_pickle(file):
    """reads the pickle file and prints it"""
    with open(file, 'rb') as f:
        biosamples_ref = pickle.load(f)
        set_df = pickle.load(f)
    print(biosamples_ref)
    print(set_df)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        read_pickle(sys.argv[1])
    else:
        print("Please provide a file to read.")
