def bin_search(data, target):
    # Set array ends.
    first = 0
    last = len(data) - 1
    
    # Find the rough middle of the array.
    while first != last:
        middle = (first + last) // 2
        
        # Check if the initial array placement is the value.
        if data[middle] == target:
            return middle
        
        # If the value is larger than the initial place:
        # Shift array up to the value after the middle, and search remainder array.
        elif data[middle] > target:
            last = middle - 1
        
        # Otherwise shift the array beginning up, and search the remainder.
        else:
            first = middle + 1

    return data.index(target)

# Set of ordered data to check.
ordered_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# Target value aimed at finding.
target = 67

# Run search for target in array.
bin_search(primes, ex_target)
