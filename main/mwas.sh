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
    echo "Please install jq by running 'sudo apt install jq'"
    exit 1
fi
if ! command -v csvjson &> /dev/null; then
    echo "Error: csvjson is not installed"
    echo "Please install csvjson by running 'sudo apt install csvkit'"
    exit 1
fi
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed"
    echo "Please install curl by running 'brew install curl' or 'sudo apt-get install curl'"
    exit 1
fi

# help menu
if [[ $1 == "-h" || $1 == "--help" || $# -eq 0 ]]; then
    cat << EOF

AAAAAAAA               AAAAAAAA  CCCCCCCC                           CCCCCCCC     GGG                  TTTTTTTTTTTTTTT
A.......A             A.......A  C......C                           C......C    G...G               TT...............T
A........A           A........A  C......C                           C......C   G.....G             T.....TTTTTT......T
A.........A         A.........A  C......C                           C......C  G.......G            T.....T     TTTTTTT
A::::::::::A       A::::::::::A   C:::::C           CCCCC           C:::::C  G:::::::::G           T:::::T
A:::::::::::A     A:::::::::::A    C:::::C         C:::::C         C:::::C  G:::::G:::::G          T:::::T
A:::::::A::::A   A::::A:::::::A     C:::::C       C:::::::C       C:::::C  G:::::G G:::::G          T::::TTTT
A::::::A A::::A A::::A A::::::A      C:::::C     C:::::::::C     C:::::C  G:::::G   G:::::G          TT::::::TTTTT
A||||||A  A||||A||||A  A||||||A       C|||||C   C|||||C|||||C   C|||||C  G|||||G     G|||||G           TTT||||||||TT
A||||||A   A|||||||A   A||||||A        C|||||C C|||||C C|||||C C|||||C  G|||||GGGGGGGGG|||||G             TTTTTT||||T
A||||||A    A|||||A    A||||||A         C|||||C|||||C   C|||||C|||||C  G|||||||||||||||||||||G                 T|||||T
A||||||A     AAAAA     A||||||A          C|||||||||C     C|||||||||C  G|||||GGGGGGGGGGGGG|||||G                T|||||T
AIIIIIIA               AIIIIIIA           CIIIIIIIC       CIIIIIIIC  GIIIIIG             GIIIIIG   TTTTTTT     TIIIIIT
AIIIIIIA               AIIIIIIA            CIIIIIC         CIIIIIC  GIIIIIG               GIIIIIG  TIIIIIITTTTTTIIIIIT
AIIIIIIA               AIIIIIIA             CIIIC           CIIIC  GIIIIIG                 GIIIIIG TIIIIIIIIIIIIIIITT
AAAAAAAA               AAAAAAAA              CCC             CCC  GGGGGGG                   GGGGGGG TTTTTTTTTTTTTTT

  Metadata Wide Association Study (MWAS)
  Version: 1.0.0


Usage: mwas [OPTIONS] [MWAS FLAGS]

input_file: path to input file
input_file format: csv, 3 columns, headers can be named anything, but must follow the format below:
  column order: accession, group, quantification
  types: string, string, int

Example:
  run,family_name,n_reads
  ERR2756783,Deltavirus,1
  ERR2756784,Bromoviridae,1
  ERR2756784,Mitoviridae,1
  ERR2756785,Totiviridae,3
  ERR2756786,Mitoviridae,4

OPTIONS:
  -h, --help                Show this help message
  -r, --run [input_file]    Run MWAS with the specified input file, without
                            downloading anything afterwards
  -rd, --run-download       Run MWAS, and download results from s3 to local
                            machine when processing is complete
  -rl, --run-log            Run MWAS, and download logs from s3 to local machine
                            when processing is complete

MWAS FLAGS: (used with -r, -rd, -rl options)
  --suppress-logging        Suppress logging (default is verbose logging)
  --no-logging              No logging (overrides --suppress-logging)
  --t-test-only             Only run t-tests (not recommended)
  --already-normalized      Input file quantifications are already normalized
                            (default is to normalize using spots from serratus's
                            logan database)
  --explicit-zeros          Quantifications of 0 are only included in the analysis
                            if they are explicitly stated in the input file
                            (default is implicit zeros)
  --p-value-threshold [FLOAT]
                            Set the p-value threshold for significance
                            (default is 0.005)
  --group-nonzero-threshold [INT]
                            Set the minimum number of non-zero quantifications
                            in a group for it to be included in the analysis
                            (default is 3; useless when --explicit-zeros is set)
  --performance-stats       Include performance statistics in the log (default
                            is to exclude them; recommended developer use only)

MWAS repository: <https://github.com/declanlim/mwas_rfam>
EOF
    exit 0
elif [[ $1 == "-r" || $1 == "--run" || $1 == "-rd" || $1 == "--run-download" || $1 == "-rl" || $1 == "--run-log" ]]; then
    # check if input file is provided
    if [[ $# -lt 2 ]]; then
        echo "Error: input file not provided"
        exit 1
    fi
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

    echo "Preparing request to send to MWAS server..."
    # get MWAS flags
    FLAGS="${@:3}"

    # jsonify the input file to send to the server (it will be reconstructed as a csv on the server)
    CSV_FILE=$2
    csvjson $CSV_FILE | jq . > request.json
    # create JSON object with data and flags
    JSON_DATA=$(jq -n --arg flags "$FLAGS" --argjson data "$(cat request.json)" '{data: $data, flags: $flags}')

    # hash the JSON_DATA to check if the request has already been made and predetermine the response's location
    hash=$(echo -n "$JSON_DATA" | md5sum | awk '{ print $1 }')
    JSON_DATA=$(echo $JSON_DATA | jq --arg hash "$hash" '. + {dest: $hash}')

    echo "Running MWAS..."
    echo "Storing MWAS output in s3 bucket: s3://serratus-biosamples/mwas_data/$hash"

    # send request to server to run MWAS (how to catch response?
    response=$(curl -s -X POST -H "Content-Type: application/json" -d "$JSON_DATA" $SERVER_URL)
    rm request.json
    # read response message and if it says with exit code 0, tell user's MWAS finished successfully otherwise, say there was an issue processing the input
    if [[ $response == *"Error"* ]]; then
        echo "Error: $response"
        exit 1
    else
        echo "MWAS finished successfully."
    fi
fi
