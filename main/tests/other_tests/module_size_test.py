"""test size of modules"""
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# Measure memory usage before import
mem_before = memory_usage()

# Import the module

import psycopg2

# Measure memory usage after import
mem_after = memory_usage()

print(f"Memory usage before import: {mem_before}")
print(f"Memory usage after import: {mem_after}")
print(f"Memory usage added by import: {mem_after - mem_before}")
