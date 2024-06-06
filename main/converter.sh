#!bin/bash

# get a list of files from the s3 bucket, and save them to a file,
# or use argument to specify a list of files to download

if [ -d "raw_metadata_csvs" ]; then
    mkdir "raw_metadata_csvs"
fi
if [ -d "converted_metadata_pickles"]; then
    mkdir "converted_metadata_pickles"
fi

S3_RAW_METADATA_DIR=s3://serratus-biosamples/bioprojects_csv/
S3_CONDENSED_METADATA_DIR=s3://serratus-biosamples/condensed-bioproject-metadata/

if [ "$1" == "-s" ]; then
    if [ "$2" ]; then
        subset=" | head -n $2"
    else
        subset=" | head -n 1000"
    fi
    size_list_file=file_list_with_sizes.txt
    cp_file_name=cp_batch_command_list.txt
    disk_files_name=pickle_files_list.txt

    s5cmd ls ${S3_RAW_METADATA_DIR} | awk '{print $NF, $(NF-1)}' ${subset} > ${size_list_file}
    echo "completed getting file list via s5cmd ls"

    while read -r line; do
        file=$(echo ${line} | awk '{print $(NF-1)}')
        echo "cp -f ${S3_RAW_METADATA_DIR}${file} ${destination}" >> ${cp_file_name}
        echo "${destination}/${file}" >> ${disk_files_name}
    done < ${size_list_file}

    echo "completed creating cp batch command list"

    s5cmd run ${cp_file_name}
    echo "completed copying raw csv files from s3 to local disk"

    time python3 converter_.py disk_files_name > log.txt
    echo "completed converting csv files to pickle files of condensed metadata"

    # sync the converted_metadata_pickles with an s3 bucket called s3://serratus-biosamples/condensed-bioproject-metadata/
    s5cmd sync -f converted_metadata_pickles/* ${S3_CONDENSED_METADATA_DIR}
    echo "completed syncing pickle files to s3"
else
    subset=""

    s5cmd cp -f ${S3_RAW_METADATA_DIR}/* ${destination}
    echo "completed copying raw csv files from s3 to local disk"

    # time python3 converter_.py ${destination} > log.txt
    time parallel -j 4 python3 converter_.py ::: ${destination}/* > log.txt
    echo "completed converting csv files to pickle files of condensed metadata"

    s5cmd sync -f converted_metadata_pickles/* ${S3_CONDENSED_METADATA_DIR}
    echo "completed syncing pickle files to s3"
fi
