#!/bin/bash

# this is what the user will run in command line to interact with MWAS:
# - provide a help menu (all flags and desc)
# - make requests to the server that hosts & runs MWAS,
# - recieve pre-processing log responses that mwas will store to s3 -> expected time, number of tests, list of bioprojects, list of ignored bioprojects
# - recieve processing log responses -> progress bar (percentage) (overwrite stdout), time remaining, number of tests completed, number of significant results found so far
# - recieve post-processing log responses -> time taken, exit status, number of significant results found, s3 link to results
# - provide a flag to download results from s3, and so, if specified, download results from s3 to local machine (results include the mwas_output as well as the log file)

SERVER_URL="http://ec2-75-101-194-81.compute-1.amazonaws.com:5000/run_mwas"
S3_BUCKET="s3://serratus-biosamples/mwas_data/"
# IMPORTANT: the arguments should be able to come in any order so we can use $num, but we can do arg then arg + 1 later for things like -r [input_file] since that DOES need to be ordered

# check dependencies: jq, csvjson (from csvkit), and prompt user to install if not found
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed"
    echo "Please install jq by running 'brew install jq' or 'sudo apt-get install jq'"
    exit 1
fi
if ! command -v csvjson &> /dev/null; then
    echo "Error: csvjson is not installed"
    echo "Please install csvjson by running 'pip install csvkit'"
    exit 1
fi
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed"
    echo "Please install curl by running 'brew install curl' or 'sudo apt-get install curl'"
    exit 1
fi

# help menu
if [[ $1 == "-h" || $1 == "--help" || $# -eq 0 ]]; then
    echo "Usage: mwas [OPTIONS] [MWAS FLAGS]"
    echo "input_file: path to input file"
    echo "input_file format: csv, 3 columns, headers can be named anything, but must follow the format below"
    echo "  column order: accession, group, quantification"
    echo "  types: string, string, int"
    echo ""
    echo "  example:"
    echo "    run,family_name,n_reads"
    echo "    ERR2756783,Deltavirus,1"
    echo "    ERR2756784,Bromoviridae,1"
    echo "    ERR2756784,Mitoviridae,1"
    echo "    ERR2756785,Totiviridae,3"
    echo "    ERR2756786,Mitoviridae,4"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help: show this help message and exit"
    echo "  -r, --run [input_file]: run MWAS with the specified input file, without downloading anything afterwards"
    echo "  -rd, --run-download [input_file]: run MWAS, and download results from s3 to local machine when processing is complete"
    echo "  -rl, --run-log [input_file]: run MWAS, and download logs from s3 to local machine when processing is complete"
    echo "MWAS FLAGS: (used with -r flag)"
    echo "  --suppress-logging: suppress logging (default is verbose logging)"
    echo "  --no-logging: no logging (overrides --suppress-logging)"
    echo "  --explicit-zeros, --explicit-zeroes: quantifications of 0 are only included in the analysis if they are explicitly stated in the input file (default is implicit zeros)"
    echo "  --combine-outputs: combine outputs of multiple bioprojects (default is to keep them separate) (presence of -d flag will enable this flag)"
    echo "  --t-test-only: only run t-tests (not recommended)"
    echo "  --already-normalized: input file quantifications are already normalized (default is to normalize using spots from serratus's logan database)"
    echo "  --p-value-threshold [FLOAT]: set the p-value threshold for significance (default is 0.005)"
    echo "  --group-nonzero-threshold [INT]: set the minimum number of non-zero quantifications in a group for it to be included in the analysis (default is 3) (useless when --explicit-zeros is set)"
    echo "  --performance-stats: include performance statistics in the log (default is to exclude them) (recommended developer use only)"
    exit 0
elif [[ $1 == "-r" || $1 == "--run" || $1 == "-rd" || $1 == "--run-download" || $1 == "-rl" || $1 == "--run-log" ]]; then
    # check if input file exists
    if [[ ! -f $2 ]]; then
        echo "Error: input file does not exist"
        exit 1
    fi
    # check if input file is a csv
    if [[ ${2: -4} != ".csv" ]]; then
        echo "Error: input file must be a csv"
        exit 1
    fi
    # check if input file has 3 columns
    if [[ $(head -n 1 $2 | tr ',' '\n' | wc -l) -ne 3 ]]; then
        echo "Error: input file must have 3 columns"
        exit 1
    fi
    # build the flags string
    FLAGS=""
    VALID_FLAGS=("--suppress-logging" "--no-logging" "--explicit-zeros" "--explicit-zeroes" "--combine-outputs" "--t-test-only" "--already-normalized" "--p-value-threshold" "--group-nonzero-threshold" "--performance-stats")
    # Loop through all command line arguments
    for arg in "$@"
    do
        if [[ " ${VALID_FLAGS[@]} " =~ " ${arg} " ]]; then
            FLAGS="$FLAGS $arg"
        fi
    done
    # jsonify the input file to send to the server (it will be reconstructed as a csv on the server)
    CSV_FILE=$2
    csvjson $CSV_FILE | jq . > request.json
    # create JSON object with data and flags
    JSON_DATA=$(jq -n --arg flags "$FLAGS" --argjson data "$(cat request.json)" '{data: $data, flags: $flags}')

    # hash the JSON_DATA to check if the request has already been made and predetermine the response's location
    hash=$(echo -n "$JSON_DATA" | md5sum | awk '{ print $1 }')
    JSON_DATA=$(echo $JSON_DATA | jq --arg hash "$hash" '. + {dest: $hash}')

    # send request to server to run MWAS (how to catch response?
    response=$(curl -X POST -H "Content-Type: application/json" -d "$JSON_DATA" $SERVER_URL)
    rm request.json
    echo $response
fi
