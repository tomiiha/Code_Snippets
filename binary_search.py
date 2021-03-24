def bin_search(data, target):
    first = 0
    last = len(ordered_set) - 1

    while first <= last:
        middle = (first + last) // 2

        if data[middle] == target:
            return middle
        elif data[middle] > target:
            last = middle - 1
        else:
            first = middle + 1

    return -1

# Set of ordered data to check

ordered_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# Target value aimed at finding.
ex_target = 67
bin_search(primes, ex_target)
