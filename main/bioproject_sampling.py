import sys
import random
import psycopg2
import pandas as pd

CONNECTION_INFO = {
    'host': 'serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com',
    'database': 'logan',
    'user': 'public_reader',
    'password': 'serratus'
}
QUERY = """
    SELECT acc as run
    FROM sra
    WHERE bioproject in (%s)
    """


def get_bioprojects_df(bioprojects: list) -> pd.DataFrame | None:
    """Get the bioproject data for the given runs
    """
    try:
        conn = psycopg2.connect(**CONNECTION_INFO)
        conn.close()  # close the connection because we are only checking if we can connect
        print(f"Successfully connected to database at {CONNECTION_INFO['host']}")
    except psycopg2.Error:
        print(f"Unable to connect to database at {CONNECTION_INFO['host']}")

    bioprojects_str = ", ".join([f"'{biopj}'" for biopj in bioprojects])

    with psycopg2.connect(**CONNECTION_INFO) as conn:
        try:
            df = pd.read_sql(QUERY % bioprojects_str, conn)
            return df
        except psycopg2.Error:
            print("Error in executing query")
            return None


class WrongFileFormatError(Exception):
    """Raised when the file format is not correct"""
    pass


if __name__ == '__main__':
    # get first arg which is a list of bioprojects
    if len(sys.argv) < 2:
        print("Usage: bioproject_sampling.py <bioprojects.txt>")
        sys.exit(1)
    elif sys.argv[1] == 'bioprojects.txt':
        # bioprojects should be a <ls -lrS> from ubuntu output of the bioprojects directory
        # i.e. -rw-rw-r-- 1 ubuntu ubuntu     3399 Jun 12 01:54 PRJNA306861.mwaspkl
        bioprojects_list = sys.argv[1]
        bioprojects_sample = []
        sizes = [1000 * 2 ** i for i in range(10)]
        size_dist = [0] * len(sizes)
        with open(bioprojects_list, 'r') as f:
            # loop through every line
            for line in f:
                vals = line.split()
                try:
                    file_size = int(vals[4])
                    bioproject_id = line.split()[8].split('.')[0]
                    bioprojects_sample.append((bioproject_id, file_size))
                except IndexError:
                    continue
        # filter out bioprojects of len 1 and 2122
        bioprojects_filtered = [x for x in bioprojects_sample if x[1] not in (1, 2122)]

        # # sort by file size
        # bioprojects_sample.sort(key=lambda x: x[1], reverse=True)
        bioprojects_sample = []
        for i in range(1, len(sizes)):
            bioprojects_in_size = [x for x in bioprojects_filtered if x[1] <= sizes[i] and x[1] > sizes[i - 1]]
            # select a random 10 bioprojects from this size
            sample = random.sample(bioprojects_in_size, 10)
            bioprojects_sample.extend(sample)
            size_dist[i - 1] = len(bioprojects_in_size)
        print(size_dist)  # [126028, 40137, 16702, 7174, 3260, 1625, 623, 283, 110, 0]
        # write to file
        with open('bioprojects_sample.txt', 'w') as f:
            for bioproject in bioprojects_sample:
                f.write(bioproject[0] + '\n')

        # query the bioprojects to make a list of runs, and then make random n_reads and group vals (synthetic)
        df = get_bioprojects_df([x[0] for x in bioprojects_sample])
        # add a column called group with random values from (A, B, C)
        df['group'] = [random.choice(['A', 'B', 'C']) for _ in range(len(df))]
        # add a column called n_reads with random values from (0, 1000)
        df['quantifier'] = [random.choice([0, 1000]) for _ in range(len(df))]
        # write to file
        df.to_csv('bioprojects_sample_mwas_test.csv', index=False)
    elif len(sys.argv) > 2 and sys.argv[1] == '-with':
        try:
            if not sys.argv[2].endswith('.txt'):
                raise WrongFileFormatError("File should be a .txt file")
            with open(sys.argv[2], 'r') as f:
                bioprojects = [line.strip() for line in f]
                df = get_bioprojects_df(bioprojects)
                df['group'] = [random.choice(['A', 'B', 'C']) for _ in range(len(df))]
                df['quantifier'] = [random.choice([0, 1000]) for _ in range(len(df))]
                df.to_csv(f'sample_mwas_test_from_{sys.argv[2][:-4]}.csv', index=False)

        except FileNotFoundError:
            print(f"File {sys.argv[2]} not found")
            sys.exit(1)
        except WrongFileFormatError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
