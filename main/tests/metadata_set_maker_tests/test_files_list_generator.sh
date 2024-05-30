#!/bin/bash
# note this script is solely meant for the aws server
if [ "$1" ]; then
        amount=$1
else
        amount=10000
fi

# save current directory
original_dir=$(pwd)

cd ~/s3_downloads/bioprojects/
ls | sort -R | head -n $amount > test_files_list_temp.txt

if [ -d test_file_list.txt ]; then
        rm test_file_list.txt
fi

touch test_file_list.txt
cd ..
mkdir -p test_csvs
cd bioprojects
while read -r line; do
        echo "/home/ubuntu/s3_downloads/bioprojects/${line}" >> test_file_list.txt
done < test_files_list_temp.txt

echo "finished listing files needed"

rm test_files_list_temp.txt
mv test_file_list.txt $original_dir

cd $original_dir
source ~/mwas_rfam/env/bin/activate
python3 unload_pickles.py test_file_list.txt /home/ubuntu/s3_downloads/test_csvs


#Failed for tube
#Failed - /home/ubuntu/s3_downloads/test_csvs/PRJNA702085.csv: Failed for tube
#Traceback (most recent call last):
#  File "/home/ubuntu/mwas_rfam/main/tests/metadata_set_maker_tests/metadata_set_maker_test.py", line 138, in single_test
#    metadata_set_maker_test_setup(file)
#  File "/home/ubuntu/mwas_rfam/main/tests/metadata_set_maker_tests/metadata_set_maker_test.py", line 60, in metadata_set_maker_test_setup
#    assert False, f"Failed for {col}"
#           ^^^^^
#AssertionError: Failed for tube
