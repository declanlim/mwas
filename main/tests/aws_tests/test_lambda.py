from mwas_lambda import lambda_handler


def test_indexed():
    indexed_event = {"bioproject_info": {"name": "PRJDB7993",
                                 "metadata_file_size": "141241",
                                 "n_biosamples": "1990",
                                 "n_sets": "1080",
                                 "n_permutation_sets": "388",
                                 "n_skippable_permutation_sets": "4",
                                 "n_groups": "20",
                                 "n_skipped_groups": "11",
                                 "num_lambda_jobs": "12",
                                 "num_conc_procs": "8",
                                 "groups": "['IFNA1', 'IFNA2', 'IFNL3', 'IFNL2', 'IFNL1', 'IFNG', 'IFNW1', 'IFNB1', 'IFNA21', 'IFNA17', 'IFNA16', 'IFNA14', 'IFNA13', 'IFNA10', 'IFNA8', 'IFNA7', 'IFNA6', 'IFNA5', 'IFNA4', 'IFNL4']"},
             "link": "Tue_Aug__6_15-20-55_2024",
             "job_window": {"IFNA1": [300, 384], "IFNW1": [0, 216]},
             "id": 1,
             "flags": {"IMPLICIT_ZEROS": "1",
                       "GROUP_NONZEROS_ACCEPTANCE_THRESHOLD": "3",
                       "ALREADY_NORMALIZED": "0",
                       "P_VALUE_THRESHOLD": "0.005",
                       "INCLUDE_SKIPPED_GROUP_STATS": "0",
                       "TEST_BLACKLISTED_METADATA_FIELDS": "0",
                       "LOGGING_LEVEL": "2",
                       "USE_LOGGER": "1",
                       "TIME_LIMIT": "60"},
            "parallel": "1"
            }
    lambda_handler(indexed_event, None)


def test_full():
    full_event = {
        "bioproject_info": {
            "name": "PRJNA136121",
            "metadata_file_size": "1723",
            "n_biosamples": "23",
            "n_sets": "10",
            "n_permutation_sets": "3",
            "n_skippable_permutation_sets": "0",
            "n_groups": "20",
            "n_skipped_groups": "0",
            "num_lambda_jobs": "1",
            "num_conc_procs": "6",
            "groups": "everything"
        },
        "link": "Tue_Aug__6_15-20-55_2024",
        "job_window": "full",
        "id": 2,
        "flags": {
            "IMPLICIT_ZEROS": "0",
            "GROUP_NONZEROS_ACCEPTANCE_THRESHOLD": "4",
            "ALREADY_NORMALIZED": "0",
            "P_VALUE_THRESHOLD": "0.005",
            "INCLUDE_SKIPPED_GROUP_STATS": "0",
            "TEST_BLACKLISTED_METADATA_FIELDS": "0",
            "LOGGING_LEVEL": "2",
            "USE_LOGGER": "1"
        }
    }
    lambda_handler(full_event, None)


def test_ttest():
    t_event = {'bioproject_info': {'name': 'PRJDB7993',
                                  'metadata_file_size': '132476',
                                  'n_biosamples': '1990',
                                  'n_sets': '1079',
                                  'n_permutation_sets': '388',
                                  'n_skippable_permutation_sets': '0',
                                  'n_groups': '20',
                                  'n_skipped_groups': '0',
                                  'num_lambda_jobs': '26',
                                  'num_conc_procs': '3',
                                  'groups': 'everything'},
              'link': 'Tue_Aug__6_15-20-55_2024',
              'job_window': 'full',
              'id': 0,
              'flags': {'IMPLICIT_ZEROS': '0',
                        'GROUP_NONZEROS_ACCEPTANCE_THRESHOLD': '4',
                        'ALREADY_NORMALIZED': '0',
                        'P_VALUE_THRESHOLD': '0.005',
                        'INCLUDE_SKIPPED_GROUP_STATS': '0',
                        'TEST_BLACKLISTED_METADATA_FIELDS': '0',
                        'LOGGING_LEVEL': '2',
                        'USE_LOGGER': '1',
                        'TIME_LIMIT': '60'}}
    lambda_handler(t_event, None)
