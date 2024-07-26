""""""
import os
import sys

ENCODING_ERRORS = {
    'too large to process': 12,
    'blacklist': 11,
    'dupe-bug': 10,
    'FAILED': 9,
    'other error': 8,
    'csv reading': 7,
    'No valid': 6,
    'invalid biosample_ids': 5,
    'Original csv was empty': 4,
    'Less than 4 rows in csv file': 3,
    'No sets were generated': 2,
    'empty': 1,
    'No issues': 0
}

DECODING_ERRORS = {
    12: 'Too large to process.',
    11: 'Blacklisted.',
    10: 'Dupe bug.',
    9: 'FAILED.',
    8: 'Other error.',
    7: 'CSV reading error.',
    6: 'All rows were scrambled.',
    5: 'Found invalid biosample_id rows.',
    4: 'Originally empty in raw file.',
    3: 'Less than 4 biosamples.',
    2: 'No sets were generated.',
    1: 'Empty file.',
    0: 'No issues.'
}


def encode(comment: str) -> int:
    """Converts a comment to a code."""
    comment_code = 0
    for msg, bit in ENCODING_ERRORS.items():
        if msg in comment:
            comment_code |= 1 << bit
    return comment_code


def decode(comment_code: int) -> str:
    """Converts a code to a comment."""
    comments = []
    for bit, msg in DECODING_ERRORS.items():
        if comment_code & (1 << bit):
            comments.append(msg)
    return ' '.join(comments)


def main(file: str):
    """csv columns: file, original_size, condensed_pickle_size, processing_time, comment, num_biosamples, num_sets, num_permutation_sets
    convert to: bioproject, n_biosamples, n_sets, n_permutation_sets, raw_md_file_size, condensed_md_file_size, comment_code"""

    with open(file, 'r') as f:
        lines = f.readlines()
    results = 'bioproject, n_biosamples, n_sets, n_permutation_sets, n_skippable_permutation_sets, raw_md_file_size, condensed_md_file_size, comment_code\n'
    skip = -1
    for i, line in enumerate(lines[1:]):
        if i == skip:
            continue
        if ', saw' in line:
            # replace the comma with whitespace for the comma that appears before the word saw
            line = line.replace(', saw', ' saw')
            skip = i + 1

        line = line.strip().split(',')
        bioproject = line[0]

        if skip == i + 1:
            n_biosamples = n_sets = n_permutation_sets = n_skippable_permutation_sets = 0
        else:
            n_biosamples = line[5]
            n_sets = line[6]
            n_permutation_sets = line[7]
            n_skippable_permutation_sets = line[8]
        raw_md_file_size = line[1]
        condensed_md_file_size = line[2]
        comment_code = encode(line[4])
        result = f'{bioproject},{n_biosamples},{n_sets},{n_permutation_sets},{n_skippable_permutation_sets},{raw_md_file_size},{condensed_md_file_size},{comment_code}\n'
        print(result)
        results += result
    with open('../data/metadata_files_info_table.csv', 'w') as f:
        f.write(results)


if __name__ == "__main__":
    if os.path.exists('../data/conversion_results.csv'):
        file = '../data/conversion_results.csv'
    elif len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        print("Please provide a file to read.")
        sys.exit(1)
    main(file)
