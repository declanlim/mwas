"""test piping"""
import random
import time
from multiprocessing import Process, Pipe

def my_function(shared_results: set, conn):
    """function"""
    random_number = random.randint(1, 100)
    if random_number in shared_results:
        print(f"we've all seen {random_number} already")
    else:
        print(f"we've never seen {random_number} before")
    time.sleep(0.000001)
    if conn:
        conn.send(random_number)
        conn.close()
    else:
        shared_results.add(random_number)

def main():
    num_instances = 60
    shared_results = set()
    num_workers = 8
    start_time = time.time()
    for batch in range(0, num_instances, num_workers):
        print(f"batch {batch}:")
        processes, parent_connections = [], []
        if num_instances - batch < num_workers:
            num_workers = num_instances - batch
        for i in range(num_workers):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            p = Process(target=my_function, args=(shared_results, child_conn))
            processes.append((p, parent_conn))
        for p, conn in processes:
            p.start()
        for p, conn in processes:
            p.join()
        for parent in parent_connections:
            shared_results.add(parent.recv())

    print(f"Piping elapsed time: {time.time() - start_time}")


    # series version to compare:

    start_time = time.time()
    for i in range(num_instances):
        my_function(shared_results, None)
    print(f"Series elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    main()
