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
CURL_SRC="http://serratus-biosamples.s3.amazonaws.com/mwas_data/"
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

  --t-test-only
  Only run t-tests (not recommended)

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

  --performance-stats
  Include performance statistics in the log
  (default is to exclude them;
  recommended developer use only)

MWAS repository:
<https://github.com/declanlim/mwas_rfam>

EOF
    fi
    exit 0
elif [[ $1 == "-ca" || $1 == "--clean-all" ]]; then
    # remove all downloaded files and directories
    rm -rf mwas_results_folder_*
    echo "All downloaded files and directories have been removed."
    exit 0
elif [[ $1 == "-r" || $1 == "--run" ]]; then
    # check dependencies: jq, csvjson (from csvkit), and prompt user to install if not found
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
    if ! command -v csvjson &> /dev/null; then
        echo "Error: csvjson is not installed"
        echo "Please install csvjson by running 'sudo apt install csvkit'"
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

    # jsonify the input file to send to the server (it will be reconstructed as a csv on the server)
    CSV_FILE=$2
    csvjson $CSV_FILE | jq . > request.json
    # create JSON object with data and flags
    JSON_DATA=$(jq -n --arg flags "$FLAGS" --arg data "$(cat request.json)" '{data: $data, flags: $flags}')
    rm request.json

    # hash the JSON_DATA to check if the request has already been made and predetermine the response's location
    hash=$(echo -n "$JSON_DATA" | md5sum | awk '{ print $1 }')
    JSON_DATA=$(echo $JSON_DATA | jq --arg hash "$hash" '. + {dest: $hash}')

    echo "================================"
    echo "      MWAS SESSION CODE:"
    echo "$hash"
    echo "================================"

    # set up directory to store results
    results_dir="$RESULTS_DIR$hash"
    if [[ ! -d $results_dir ]]; then
        mkdir $results_dir
    fi

    # send request to server to run MWAS (how to catch response?
    response=$(curl -s -X POST -H "Content-Type: application/json" -d "$JSON_DATA" $SERVER_URL)

    # read response message and if it says with exit code 0, tell user's MWAS finished successfully otherwise, say there was an issue processing the input
    message=$(echo $response | jq -r '.message')
    status=$(echo $response | jq -r '.status')
    error=$(echo $response | jq -r '.error')

    # check if the response contains an error message
    if [[ $error != "null" ]]; then
        echo "Error: $error"
        exit 1
    else
        echo "$message"
        echo "$status"
    fi
elif [[ $1 == "-g" || $1 == "--get" ]]; then
    # check if session code is provided
    if [[ $# -lt 2 ]]; then
        echo "Error: session code not provided"
        exit 1
    fi

    # get session code
    SESSION_CODE=$2

    #  directory to store results
    results_dir="$RESULTS_DIR$SESSION_CODE"
    if [[ ! -d $results_dir ]]; then
        mkdir $results_dir
    fi
    cd $results_dir

    # check if session code exists via progress report since that is always available if running
    report=$(curl -s -o progress_report.json $CURL_SRC$SESSION_CODE/progress_report.json)
    DNE=$(cat progress_report.json | grep "The specified key does not exist" | wc -l)
    if [[ $DNE -eq 1 ]]; then
        echo "Error: session code does not exist"
        rm progress_report.json
        cd ..
        rmdir $results_dir
        exit 1
    fi

    if [[ $3 == "-p" || $3 == "--progress" || $3 == "-vp" || $3 == "--verbose-progress" ]]; then
        # progress
        report=$(jq '.' progress_report.json)
        printf '=%.0s' $(seq 1 $(tput cols))
        echo "MWAS PROGRESS REPORT:"
        echo "Status: $(echo $report | jq -r '.status' 2>/dev/null)"
        echo "elapsed time: $(echo $report | jq -r '.time_elapsed' 2>/dev/null)"
        if [[ $3 == "-vp" || $3 == "--verbose-progress" ]]; then
            # verbose progress
            echo "Start time: $(echo $report | jq -r '.start_time' 2>/dev/null)"
            echo "Number of aws lambdas: $(echo $report | jq -r '.num_lambdas_jobs' 2>/dev/null)"
            echo "Number of aws lambdas completed: $(echo $report | jq -r '.num_jobs_completed' 2>/dev/null)"
            echo "Number of permutation tests: $(echo $report | jq -r '.num_permutation_tests' 2>/dev/null)"
            echo "total cost: $(echo $report | jq -r '.total_cost' 2>/dev/null)"
            successes=$(echo $report | jq -r '.successes' 2>/dev/null)
            fails=$(echo $report | jq -r '.fails' 2>/dev/null)
            rate=$(echo "scale=2; $successes / ($successes + $fails) * 100" | bc)
            echo "Success rate: ${rate}%"

        fi
        printf '=%.0s' $(seq 1 $(tput cols))
    fi

    if [[ $3 == "-l" || $3 == "--log" || $3 == "-a" || $3 == "--all" ]]; then
        # download log file
        curl -o mwas_log.txt -s $CURL_SRC$SESSION_CODE/mwas_logging.txt
        failed=$(cat mwas_log.txt | grep "The specified key does not exist" | wc -l)
        if [[ $failed -eq 1 ]]; then
            echo "Error: There was no log file."
        else
            echo "Log file downloaded."
        fi
    fi
    if [[ $3 == "-ib" || $3 == "--ignored-biopjs" || $3 == "-a" || $3 == "--all" ]]; then
        # download ignored bioprojects file
        curl -o ignored_bioprojects.txt -s $CURL_SRC$SESSION_CODE/problematic_biopjs.txt
        failed=$(cat ignored_bioprojects.txt | grep "The specified key does not exist" | wc -l)
        if [[ $failed -eq 1 ]]; then
            echo "Error: There was no ignored bioprojects list."
        else
            echo "Ignored_bioprojects list downloaded."
        fi
    fi
    if [[ $3 == "-o" || $3 == "--output" || $3 == "-a" || $3 == "--all" || $3 == "" ]]; then
        # check if it's already downloaded
        if [[ -f mwas_output.csv ]]; then
            echo "MWAS output was already downloaded."
        else
            # download output file
            curl -o mwas_output.csv -s $CURL_SRC$SESSION_CODE/mwas_output.csv
            # check if success
            failed=$(cat mwas_output.csv | grep "The specified key does not exist" | wc -l)
            if [[ $failed -eq 1 ]]; then
                echo "Error: MWAS is still running. Could not download results yet."
            else
                echo "MWAS output downloaded."
            fi
        fi
    fi
    cd ..
fi
