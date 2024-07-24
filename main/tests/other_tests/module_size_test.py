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

import math
# built-in libraries
import os
import sys
import platform
import subprocess
import pickle
import time
import tracemalloc
import logging
import json
from atexit import register
from shutil import rmtree
from typing import Any

import boto3
# import required libraries
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import permutation_test, ttest_ind_from_stats

# Measure memory usage after import
mem_after = memory_usage()

print(f"Memory usage before import: {mem_before}")
print(f"Memory usage after import: {mem_after}")
print(f"Memory usage added by import: {mem_after - mem_before}")
