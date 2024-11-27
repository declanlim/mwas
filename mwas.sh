#!/bin/bash

# this is what the user will run in command line to interact with MWAS:
# - provide a help menu (all flags and desc)
# - make requests to the server that hosts & runs MWAS,
# - recieve pre-processing log responses that mwas will store to s3 -> expected time, number of tests, list of bioprojects, list of ignored bioprojects
# - recieve processing log responses -> progress bar (percentage) (overwrite stdout), time remaining, number of tests completed, number of significant results found so far
# - recieve post-processing log responses -> time taken, exit status, number of significant results found, s3 link to results
# - provide a flag to download results from s3, and so, if specified, download results from s3 to local machine (results include the mwas_output as well as the log file)

SERVER_URL="http://ec2-75-101-194-81.compute-1.amazonaws.com:5000/run_mwas"
S3_BUCKET="mwas-user-dump"
RESULTS_DIR="mwas_results_folder_"
# IMPORTANT: the arguments should be able to come in any order so we can use $num, but we can do arg then arg + 1 later for things like -r [input_file] since that DOES need to be ordered

# help menu
if [[ $1 == "-h" || $1 == "--help" || $# -eq 0 ]]; then
    if [[ $(tput cols) -gt 118 ]]; then
        cat << EOF

AAAAAAAA               AAAAAAAA  CCCCCCCC                           CCCCCCCC     GGG                  TTTTTTTTTTTTTTT
A.......A             A.......A  C......C                           C......C    G...G               TT...............TT
A........A           A........A  C......C                           C......C   G.....G             T..................T
A.........A         A.........A  C......C                           C......C  G.......G            T......TTTTTTT.....T
A::::::::::A       A::::::::::A   C:::::C           CCCCC           C:::::C  G:::::::::G           T:::::T      TTTTTTT
A:::::::::::A     A:::::::::::A    C:::::C         C:::::C         C:::::C  G:::::G:::::G          T:::::T
A:::::::A::::A   A::::A:::::::A     C:::::C       C:::::::C       C:::::C  G:::::G G:::::G         T::::::TTTTTTT
A::::::A A::::A A::::A A::::::A      C:::::C     C:::::::::C     C:::::C  G:::::G   G:::::G         TT:::::::::::TTT
A||||||A  A||||A||||A  A||||||A       C|||||C   C|||||C|||||C   C|||||C  G|||||G     G|||||G          TTT|||||||||||TT
A||||||A   A|||||||A   A||||||A        C|||||C C|||||C C|||||C C|||||C  G|||||GGGGGGGGG|||||G            TTTTTTT||||||T
A||||||A    A|||||A    A||||||A         C|||||C|||||C   C|||||C|||||C  G|||||||||||||||||||||G                  T|||||T
A||||||A     AAAAA     A||||||A          C|||||||||C     C|||||||||C  G|||||GGGGGGGGGGGGG|||||G    TTTTTTT      T|||||T
AIIIIIIA               AIIIIIIA           CIIIIIIIC       CIIIIIIIC  GIIIIIG             GIIIIIG   TIIIIITTTTTTTTIIIIIT
AIIIIIIA               AIIIIIIA            CIIIIIC         CIIIIIC  GIIIIIG               GIIIIIG  TIIIIIIIIIIIIIIIIIIT
AIIIIIIA               AIIIIIIA             CIIIC           CIIIC  GIIIIIG                 GIIIIIG TTIIIIIIIIIIIIIIITT
AAAAAAAA               AAAAAAAA              CCC             CCC  GGGGGGG                   GGGGGGG  TTTTTTTTTTTTTTT

EOF
    else
        cat << EOF

 __  ____          __      _____
|  \/  \ \        / /\    / ____|
| \  / |\ \  /\  / /  \  | (___
| |\/| | \ \/  \/ / /\ \  \___ \\
| |  | |  \  /\  / ____ \ ____) |
|_|  |_|   \/  \/_/    \_\_____/

EOF
    fi
    if [[ $(tput cols) -gt 82 ]]; then
        cat << EOF
Metadata Wide Association Study (MWAS)
Version: 1.0.0


Usage: mwas [MODE] [ARGUMENTS]

MWAS_FLAGS:   flags to customize the MWAS run. See below for available flags.
SESSION_CODE: Unique code provided by MWAS server when starting MWAS.
              Used to download results, check progress, etc.
INPUT_FILE:   path to input file. Format: CSV, with 3 columns.
              headers can be named anything, but must follow the format below:
              csv column order: accession, group, quantification
              column types: STRING, STRING or INT, INT

              Example:
                run,family_name,n_reads
                ERR2756783,Deltavirus,1
                ERR2756784,Bromoviridae,1
                ERR2756784,Mitoviridae,1
                ERR2756785,Totiviridae,3
                ERR2756786,Mitoviridae,4

MODE:
  -h, --help                display this help menu and exit
  -r, --run [INPUT_FILE] [MWAS_FLAGS]...
                            run MWAS with the specified input file, without
                            downloading anything afterwards. Provides a session
                            code relative to your input
  -g, --get [SESSION_CODE] [OPTIONS]...
                            download results from MWAS run, where [SESSION_CODE]
                            is the code provided from running MWAS.
                            [OPTIONS] specifies what to download. If no options
                            are provided, it will default to the --output option.
  -rg, --run-get [INPUT_FILE] [MWAS_FLAGS]...
                            run MWAS with the specified input file, displays progress
                            report, and downloads the results afterwards.
                            Provides a session code relative to your input.
  -ca, --clean-all          Remove all downloaded files and directories

OPTIONS: (used with -g, --get)
  -o, --output              get the MWAS output results csv file
                            (will not return anything if MWAS is still running)
  -l, --log                 get the MWAS run logs
                            (verbosity depends on the MWAS_FLAGS you set)
  -ib, --ignored-biopjs     Download the list of ignored bioprojects
  -p, --progress            Report progress of an MWAS run. Will report percentage
                            of completion, elapsed_time and time_remaining
  -vp, --verbose-progress   Report progress of an MWAS run in verbose mode
                            (provides more detailed information than --progress)
  -a, --all                 Download everything associated with the MWAS run
                            (output, log, progress report, and ignored bioprojects)

MWAS_FLAGS: (used with -r, --run)
  --suppress-logging        Suppress logging (default is verbose logging)
  --no-logging              No logging (overrides --suppress-logging)
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

MWAS repository: <https://github.com/declanlim/mwas>

EOF
    else
      # small logo to fit in smaller terminal windows
      cat << EOF
Metadata Wide Association Study (MWAS)
Version: 1.0.0

Usage: mwas [MODE] [ARGUMENTS]

MWAS_FLAGS:
flags to customize the MWAS run.
See below for available flags.

SESSION_CODE:
Unique code provided by MWAS server
when starting MWAS.
Used to download results, check progress, etc.

INPUT_FILE:
path to input file. Format: CSV, with 3 columns.
headers can be named anything,
but must follow the format below:
csv column order: accession, group, quantification
column types: STRING, STRING or INT, INT

Example:
  run,family_name,n_reads
  ERR2756783,Deltavirus,1
  ERR2756784,Bromoviridae,1
  ERR2756784,Mitoviridae,1
  ERR2756785,Totiviridae,3
  ERR2756786,Mitoviridae,4

MODE:
  -h, --help
  display this help menu and exit

  -r, --run [INPUT_FILE] [MWAS_FLAGS]...
  run MWAS with the specified input file,
  without downloading anything afterwards.
  Provides a session code relative
  to your input

  -g, --get [SESSION_CODE] [OPTIONS]...
  download results from MWAS run,
  where [SESSION_CODE] is the code provided
  from running MWAS. [OPTIONS] specifies what
  to download. If no options are provided,
  it will default to the --output option.

  -ca, --clean-all
  Remove all downloaded files and directories

OPTIONS: (used with -g, --get)
  -o, --output
  get the MWAS output results csv file
  (will not return anything
  if MWAS is still running)

  -l, --log
  get the MWAS run logs
  (verbosity depends on the
  MWAS_FLAGS you set)

  -ib, --ignored-biopjs
  Download the list of ignored bioprojects

  -p, --progress
  Report progress of an MWAS run.
  Will report percentage of completion,
  elapsed_time and time_remaining

  -vp, --verbose-progress
  Report progress of an MWAS run
  in verbose mode (provides more detailed
  information than --progress)

  -a, --all
  Download everything associated
  with the MWAS run (output, log,
  progress report, and ignored bioprojects)

MWAS_FLAGS: (used with -r, --run)
  --suppress-logging
  Suppress logging (default is verbose logging)

  --no-logging
  No logging (overrides --suppress-logging)

  --already-normalized
  Input file quantifications are already
  normalized (default is to normalize using
  spots from serratus's logan database)

  --explicit-zeros
  Quantifications of 0 are only included
  in the analysis if they are explicitly stated
  in the input file (default is implicit zeros)

  --p-value-threshold [FLOAT]
  Set the p-value threshold for significance
  (default is 0.005)

  --group-nonzero-threshold [INT]
  Set the minimum number of non-zero
  quantifications in a group for it to be
  included in the analysis (default is 3;
  useless when --explicit-zeros is set)

MWAS repository:
<https://github.com/declanlim/mwas>

EOF
    fi
    exit 0
elif [[ $1 == "-ca" || $1 == "--clean-all" ]]; then
    # remove all downloaded files and directories
    rm -rf mwas_run_folder_*
    echo "All downloaded files and directories have been removed."
    exit 0
elif [[ $1 == "-r" || $1 == "--run" ]]; then
    # check dependencies: jq, and prompt user to install if not found
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is not installed"
        echo "Please install curl by running 'brew install curl' or 'sudo apt-get install curl'"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is not installed"
        echo "Please install jq by running 'sudo apt install jq'"
        exit 1
    fi
    if ! command -v sha256sum &> /dev/null; then
        echo "Error: sha256sum is not installed"
        echo "Please install sha256sum by running 'sudo apt install sha256sum'"
        exit 1
    fi

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
    FLAGS_HASH=$(echo -n "$FLAGS" | sha256sum | awk '{ print $1 }')
    # get CSV file hash
    CSV_FILE=$2
    FILE_HASH=$(sha256sum $CSV_FILE | awk '{ print $1 }')
    # hash value used to reference this mwas run
    hash=$(echo -n "${FILE_HASH}${FLAGS_HASH}" | sha256sum | awk '{ print $1 }')

     # data should be have a hash attribute and flags attribute, where flags is a dict of the flags and their values (e.g. true or false or a value for thinsg like p-value setting)
    JSON_DATA="{\"hash\": \"$hash\", \"flags\": {"
    # if flags is empty
    if [[ ${#FLAGS} -eq 0 ]]; then
      JSON_DATA="$JSON_DATA}}"
    else
      # split flags into an array
      FLAGS=($FLAGS)
      for ((i=0; i<${#FLAGS[@]}; i++)); do
        flag=${FLAGS[$i]}
        case $flag in
          --suppress-logging)
            JSON_DATA="$JSON_DATA\"suppress_logging\": 1,"
            ;;
          --no-logging)
            JSON_DATA="$JSON_DATA\"no_logging\": 1,"
            ;;
          --ignore-dynamo)
            JSON_DATA="$JSON_DATA\"ignore_dynamo\": 1,"
            ;;
          --already-normalized)
            JSON_DATA="$JSON_DATA\"already_normalized\": 1,"
            ;;
          --explicit-zeros|--explicit-zeroes)
            JSON_DATA="$JSON_DATA\"explicit_zeros\": 1,"
            ;;
          --p-value-threshold)
            # get the value of the next flag
            next_flag=${FLAGS[$i+1]}
            # check if it's a float between 0 and 1
            if [[ ! $next_flag =~ ^0.[0-9]+$ ]]; then
                echo "Error: p-value threshold must be a float between 0 and 1"
                exit 1
            fi
            JSON_DATA="$JSON_DATA\"p_value_threshold\": $next_flag,"
            ;;
          --group-nonzero-threshold)
            next_flag=${FLAGS[$i+1]}
            # check if it's an integer
            if [[ ! $next_flag =~ ^[0-9]+$ ]]; then
                echo "Error: group nonzero threshold must be an integer"
                exit 1
            fi
            JSON_DATA="$JSON_DATA\"group_nonzero_threshold\": $next_flag,"
            ;;
        esac
      done
      JSON_DATA="${JSON_DATA::-1}}}"  # remove last comma and close the dict
    fi
    echo "JSON_DATA: $JSON_DATA"

    # get presigned url for the input file
    echo "getting presigned url to upload input file..."
    API_GATEWAY_URL="https://6fnk2z2hcj.execute-api.us-east-1.amazonaws.com"
    # needs attributes bucket_name and hash
    RESPONSE=$(curl -sS -X POST $API_GATEWAY_URL/presigned_url_generator \
             -H "Content-Type: application/json" \
             -d "{\"bucket_name\": \"$S3_BUCKET\", \"hash\": \"$hash\"}")
    # check status
    STATUS=$(echo $RESPONSE | jq -r '.statusCode')
    if [[ $STATUS != 200 ]]; then
        # read error message in body
        ERROR=$(echo $RESPONSE | jq -r '.message')
        echo "Error: $ERROR"
        exit 1
    fi
    PRESIGNED_URL=$(echo $RESPONSE | jq -r '.presigned_url')
    HASH=$(echo $RESPONSE | jq -r '.hash')
    # verify that the hash is the same as the one we calculated
    if [[ $HASH != $hash ]]; then
        echo "Error: Hashes do not match"
        exit 1
    fi

    # using the presigned url, upload the file (it will go to the correct folder in s3)
    # save name of file
    echo "Uploading your input file to s3..."
    new_name="input.csv"
    if [[ $CSV_FILE != $new_name ]]; then
        mv $CSV_FILE $new_name
    fi
    curl -v -sS --upload-file $new_name $PRESIGNED_URL
    # check if the file was uploaded successfully
    if [[ $? -ne 0 ]]; then
        echo "Error: There was an issue uploading the input file to s3"
        exit 1
    fi
    # restore name
    if [[ $CSV_FILE != $new_name ]]; then
        mv $new_name $CSV_FILE
    fi

    # set up directory to store results
    results_dir="mwas_run_folder_$hash"
    if [[ ! -d $results_dir ]]; then
        mkdir $results_dir
    fi
    echo "set up a local dir for you: $results_dir"

    # send API request to start MWAS
    echo "sending $JSON_DATA to the mwas preprocessing aws lambda to begin the mwas run"
    echo "================================"
    echo "      MWAS SESSION CODE:"
    echo "$hash"
    echo "================================"

    RESPONSE=$(curl -sS -X POST $API_GATEWAY_URL/mwas_initiate \
             -H "Content-Type: application/json" \
             -d "$JSON_DATA")
    STATUS=$(echo $RESPONSE | jq -r '.statusCode')
    if [[ $STATUS != 200 ]]; then
        # read error message in body
        ERROR=$(echo $RESPONSE | jq -r '.message')
        if [[ $ERROR == "" ]]; then
            echo "Since we haven't gotten an error yet, things should be running fine."
            exit 0
        fi
        echo "ERROR: $ERROR"
        exit 1
    fi
elif [[ $1 == "-g" || $1 == "--get" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Error: session code not provided"
        exit 1
    fi
    echo "checking progress for your MWAS run..."

    # download progress.json from s3
    SESSION_CODE=$2
    curl -s -H "Host: ${S3_BUCKET}.s3.amazonaws.com" \
         -H "Content-Type: application/json" \
         https://${S3_BUCKET}.s3.amazonaws.com/${SESSION_CODE}/progress.json \
         -o progress_file.json

    # read progress.json
    STATUS=$(jq -r '.status' progress_file.json)
    if [[ $STATUS != "MWAS COMPLETED" ]]; then
        echo "MWAS is still running. Current status: $STATUS"
        rm progress_file.json
        exit 0
    fi
    echo "MWAS has completed. Downloading results..."
    # now download
    curl -s -H "Host: ${S3_BUCKET}.s3.amazonaws.com" \
         -H "Content-Type: application/json" \
         https://${S3_BUCKET}.s3.amazonaws.com/${SESSION_CODE}/mwas_output.csv \
         -o mwas_output_${SESSION_CODE}.csv
    # delete progress file
    rm progress_file.json
    exit 0
else
    echo "Error: invalid flag. Run 'mwas -h' for help"
    exit 1
fi
