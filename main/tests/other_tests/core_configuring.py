
LAMBDA_SIZES = ((2, 900), (3, 3600), (4, 5400), (5, 7200), (6, 10240))

def main(b: int) -> tuple[int, int]:
    """getting mem and core count for lambda function"""
    # choosing lambda size
    mem_width = b * 240000 + 84 * 1024 ** 2  # this is in bytes
    # note, num of cpu cores = index in LAMBDA_SIZES + 2, so e.g. 2 cores for 900MB, 3 cores for 3600MB, etc.
    size_index = len(LAMBDA_SIZES) - 1  # start with the largest size, since it has better compute power and more cores
    n_conc_procs = LAMBDA_SIZES[size_index][0]
    while mem_width * n_conc_procs * 10.5 < (LAMBDA_SIZES[size_index][1] * 1024 ** 2) * 0.8:
        if size_index == 0:
            break
        size_index -= 1
        n_conc_procs = LAMBDA_SIZES[size_index][0]
    while mem_width * n_conc_procs > (LAMBDA_SIZES[size_index][1] * 1024 ** 2) * 0.8:
        n_conc_procs -= 1
        if n_conc_procs < 1:
            size_index -= 1
            if size_index == -1:
                print(f"Error: not enough memory to run a single test on a lambda functions for bio_project")
                return 0, 0
            n_conc_procs = LAMBDA_SIZES[size_index][0]

    print(f"Chosen lambda size: {LAMBDA_SIZES[size_index][1]}MB, {n_conc_procs} concurrent processes")
    return LAMBDA_SIZES[size_index][1], n_conc_procs


if __name__ == '__main__':
    for i in range(10, 50000, 100):
        print(i)
        main(i)
