"""Lambda function - for statistical tests - for MWAS"""

from mwas_functions import *

logger = None


def lambda_handler(event: dict, context: Any) -> dict:
    """Lambda handler for MWAS"""
    # get the data from event
    bioproject_info = event['bioproject_info']
    main_df_link = event['main_df_link']
    job_window = event['job_window']
    id = event['id']
    FLAGS = event['FLAGS']
    # update flags
    for flag in FLAGS:
        globals()[flag] = FLAGS[flag]

    # create logger
    global logger
    logger = logging.getLogger(f"log_{bioproject_info['name']}_job{id}.txt")
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(f"log_{bioproject_info['name']}_job{id}.txt")
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # get the main_df
    try:
        command_cp = f"s5cmd cp -f {main_df_link} main_df.pickle"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        with open('main_df.pickle', 'rb') as f:
            main_df = pickle.load(f)
        os.remove('main_df.pickle')

    except Exception as e:
        log_print(f"Error in loading main_df: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in loading main_df: {e}"}

    # get the bioproject info
    bioproject = BioProjectInfo(**bioproject_info)
    bioproject.retrieve_metadata()
    result = bioproject.process_bioproject(main_df, job_window, id)
    if result is None:
        return {'statusCode': 500, 'body': f"Error in processing bioproject {bioproject.name} job {id}"}
    # make result file
    with open('result.txt', 'w') as f:
        f.write(result)

    # upload the result and log files to s3
    try:
        command_cp = f"s5cmd cp -f result.txt {S3_OUTPUT_DIR}/{OUTPUT_FILES_DIR}/result_{bioproject.name}_job{id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove('result.txt')

        command_cp = f"s5cmd cp -f log_{bioproject.name}_job{id}.txt {S3_OUTPUT_DIR}/{PROC_LOG_FILES_DIR}/log_{bioproject.name}_job{id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove(f"log_{bioproject.name}_job{id}.txt")

    except Exception as e:
        log_print(f"Error in syncing output: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in putting output to s3: {e}"}

    return {'statusCode': 200, 'body': f'MWAS processing completed for {bioproject.name} job {id}'}
