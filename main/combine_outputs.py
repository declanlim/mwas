"""combines MWAS outputs by a date - if no date specified, choose most recent date"""

import os
import sys
import re


OUTPUT_DIR_DISK = 'outputs'
MONTH_TO_NUM = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python combine_outputs.py <date>')
        print('e.g. date = Fri_Jun_28_22-34-02_2024')
        print('or date = most_recent')
        sys.exit(1)

    date = sys.argv[1]
    if date == 'most_recent':
        # find the most recent date
        # filter if they match the regex like so
        dates = [d.split('_output')[-1].split('.')[0] for d in os.listdir(OUTPUT_DIR_DISK) if re.match(r'^PRJ\w{2}\d*_output_\w{3}_\w{3}_\d{2}_\d{2}-\d{2}-\d{2}_\d{4}.csv$', d)]
        # dates are in the form of 'Fri_Jun_28_22-34-02_2024' so get max is tricky:
        # convert to a tuple of (year, month, day, hour, minute, second) and compare
        # note the - and _ in the date

        def date_key(date):
            date = date.split('_')
            time = date[-2].split('-')
            return (int(date[-1]), MONTH_TO_NUM[date[-4]], int(date[-3]), int(time[0]), int(time[1]), int(time[2]))
        date = max(dates, key=date_key)
        print(f'Most recent date: {date}')

    for file in os.listdir(OUTPUT_DIR_DISK):
        if date not in file:
            continue

        # make sure the file's date is the same as the date mwas started
        try:
            if file.endswith('.csv') and file.split('_output')[-1].split('.')[0] == date:
                with open(f"{OUTPUT_DIR_DISK}/{file}", 'r') as f:
                    with open(f"{OUTPUT_DIR_DISK}/combined_output_{date}.csv", 'a') as combined:
                        combined.write(f.read())
                os.remove(f"{OUTPUT_DIR_DISK}/{file}")
        except Exception as e:  # string splitting error or file not found
            print(f"Error while combining output files with: {e}")
            continue
