import secrets
import string
import time
from multiprocessing import Process, cpu_count
from threading import Lock
import hashlib
import boto3

file_lock = Lock()
file_dir = ""  # "/tmp/" if running on lambda

def write_to_file(file_name: str, data: str):
    """write data to file"""
    with file_lock:
        with open(file_name, 'a') as f:
            f.write(data)  # note data will be a string with a newline at the end, so no need to add it here

def task(raw_value: str):
    """simulate a cpu intensive task (sleep will not simulate parallelism properly)"""
    print("Task started for value: ", raw_value)
    hashed_value = hashlib.pbkdf2_hmac('sha256', raw_value.encode(), 'salty-salt'.encode(), 100000)
    print("Task finished")
    return hashed_value

def task_batch(batch: list, file: str):
    """task for batch"""
    hashed_value_str = ""
    for password in batch:
        hashed_value = task(password)
        hashed_value_str += str(password) + "\n"
    write_to_file(file, hashed_value_str)
    return "success"

def lambda_handler(event: dict, context):
    """Lambda handler for hash test"""
    num_items = event['num_items']
    num_workers = cpu_count()
    print(f"num_workers: {num_workers}")
    passwords = []
    times = []

    # generate passwords
    alphabet = string.ascii_letters + string.digits
    for _ in range(num_items):
        passwords.append(''.join(secrets.choice(alphabet) for i in range(24)))

    # create output_series_parallel.txt and output_series.txt in tmp folder
    series_parallel_file = f"{file_dir}output_series_parallel.txt"
    series_file = f"{file_dir}output_series.txt"
    with open(series_parallel_file, "w") as f:
        f.write("")
    with open(series_file, "w") as f:
        f.write("")

    # ========================
    # multithreading series-parallel (divide the work in batches so each thread works once on a batch)
    time_s = time.perf_counter()
    batch_size = num_items // num_workers
    workers = {}
    for i in range(0, num_workers):
        batch = passwords[i * batch_size: (i + 1) * batch_size]
        workers[i] = Process(target=task_batch, args=(batch, series_parallel_file))
        workers[i].start()
    for i in range(num_workers):
        workers[i].join()
    time_e = time.perf_counter()
    print(f"Elapsed time series-parallel: {time_e - time_s}")
    times.append(time_e - time_s)

    # ========================
    # series compare:

    time_s = time.perf_counter()
    hashed_value_str = ""
    for password in passwords:
        hashed_value = task(password)
        hashed_value_str += str(password) + "\n"
    write_to_file(series_file, hashed_value_str)
    time_e = time.perf_counter()
    print(f"Elapsed time series: {time_e - time_s}")
    times.append(time_e - time_s)

    print(f"Elapsed times: \n series-parallel: {times[0]}, series: {times[1]}")
    print(f"num_workers: {num_workers}")

    # store the results in an s3 bucket
    try:
        s3 = boto3.client('s3')
        s3.upload_file(series_parallel_file, "serratus-biosamples", "mwas_lambda_zips/" +
                       (series_parallel_file[5:] if file_dir == "/tmp/" else series_parallel_file))
        print(f"Uploaded {series_parallel_file}")
    except Exception as e:
        print(f"Error in uploading file: {e}")


if __name__ == '__main__':
    indexed_event = {"num_items": 40}
    lambda_handler(indexed_event, None)
