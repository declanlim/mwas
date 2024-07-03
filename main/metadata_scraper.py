"""collects all necessary info from all the condensed metadata files on s3 to produce a dataset for a database to be utilized by MWAS
also, gets a list of all the possible metadata fields and their types for the database"""

import os

S3_METADATA_DIR = 's3://serratus-biosamples/condensed-bioproject-metadata'

# def main():
#     """"""
