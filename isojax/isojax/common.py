def largest_divisible_core(divisor, ncore=10):
    largest_divisible = None
    for num in range(1, ncore+ 1):
        if divisor % num == 0:
            if largest_divisible is None or num > largest_divisible:
                largest_divisible = num
    return largest_divisible
