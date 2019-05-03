import re
def read_fds(result_path):
    # Reads a metanome result-file at location result_path.
    # Returns a dictionary with funtionally determined column as key and
    # arrays of functionally determining column-combinations as values.
    fd_dict = dict()
    save_fds = False
    with open(result_path) as f:
        for line in f:
            if save_fds:
                line = re.sub('\n', '', line)
                splits = line.split("->")

                # Convert to int
                splits[0] = [int(x) for x in splits[0].split(',')]
                splits[1] = int(splits[1])

                if splits[1] in fd_dict:
                    fd_dict[splits[1]].append(splits[0])
                else:
                    fd_dict[splits[1]] = [splits[0]]

            if line == '# RESULTS\n': # Start saving FDs
                save_fds = True

    return fd_dict

result_path = 'performance-measure/results/HyFD-1.2-SNAPSHOT.jar2019-04-24T163826_fds'
fds = read_fds(result_path)
