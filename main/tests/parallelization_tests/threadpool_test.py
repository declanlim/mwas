"""parallel processing with ThreadPoolExecutor (multithreading)
source (starting point): https://medium.com/@jvroig/multithreading-in-aws-lambda-part-2-multithreaded-code-in-python-d45f11bae155
note - I found that a combination of series with parallel processing is the most efficient for their use case, at least with my (reasonable number of cores)

for example, with 240 items I got
Elapsed times:
 parallel: 70.9726300999755, series-parallel: 10.199080399936065, series: 18.397091299993917

 likely due to the overheard times of starting and stopping threads. So, series parallel only has the overhead for num_workers threads, while parallel has the overhead for all 240 threads.
 note that possibly, for small numbers of items, series is the best option, as the overhead of starting threads is not worth it.

 e.g. around 40 items, series is slightly better than series-parallel, so it's best to find an equation that uses time of task, number of items and number of workers to determine the best method

 this equation is: time_series = time_task * num_items, time_series_parallel = overhead * num_workers + time_task * num_items / num_workers, time_parallel = overhead * num_items + time_task * num_items
"""
import secrets
import string
import time
from multiprocessing import Process, cpu_count
import hashlib


def task(raw_value: str):
    """simulate a cpu intensive task (sleep will not simulate parallelism properly)"""
    print("Task started for value: ", raw_value)
    hashed_value = hashlib.pbkdf2_hmac('sha256', raw_value.encode(), 'salty-salt'.encode(), 100000)
    print("Task finished")
    return hashed_value

def task_batch(batch: list):
    """task for batch"""
    for password in batch:
        task(password)
    return "hello"

def lambda_handler(event: dict, context):
    """Lambda handler for hash test"""
    num_items = 40
    num_workers = cpu_count()
    print(f"num_workers: {num_workers}")
    workers = {}
    passwords = []
    times = []

    # generate passwords
    alphabet = string.ascii_letters + string.digits
    for _ in range(num_items):
        passwords.append(''.join(secrets.choice(alphabet) for i in range(24)))

    time_s = time.perf_counter()

    # multithreading
    ctr = 0
    for password in passwords:
        if ctr < num_workers:
            workers[ctr] = Process(target=task, args=(password,))
            workers[ctr].start()
        else:
            # join workers (wait for all active workers to finish)
            for i in range(num_workers):
                workers[i].join()
            # workers are finished (start another batch)
            ctr = 0
            workers[ctr] = Process(target=task, args=(password,))
            workers[ctr].start()
        ctr += 1

    for i in range(num_workers):
        workers[i].join()

    time_e = time.perf_counter()
    print(f"Elapsed time parallel: {time_e - time_s}")
    times.append(time_e - time_s)

    # ========================
    # multithreading series-parallel (divide the work in batches so each thread works once on a batch)
    time_s = time.perf_counter()
    batch_size = num_items // num_workers
    workers = {}
    for i in range(0, num_workers):
        batch = passwords[i * batch_size: (i + 1) * batch_size]
        workers[i] = Process(target=task_batch, args=(batch,))
        workers[i].start()
    for i in range(num_workers):
        workers[i].join()
        # print(f"Worker {i} finished with return string {workers[i].returncode}")
    time_e = time.perf_counter()
    print(f"Elapsed time series-parallel: {time_e - time_s}")
    times.append(time_e - time_s)

    # ========================
    # series compare:

    time_s = time.perf_counter()
    for password in passwords:
        task(password)
    time_e = time.perf_counter()
    print(f"Elapsed time series: {time_e - time_s}")
    times.append(time_e - time_s)

    print(f"Elapsed times: \n parallel: {times[0]}, series-parallel: {times[1]}, series: {times[2]}")


if __name__ == '__main__':
    lambda_handler({}, None)
