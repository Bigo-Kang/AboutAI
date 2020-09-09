import math


def is_prime(number):
    if number <= 1:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for div in range(3, int(math.sqrt(number) + 1), 2):
        if number % div == 0:
            return False
    return True


def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1


prime_iterator = get_primes(1)

for _ in range(100):
    next_prime_number = next(prime_iterator)
    print(next_prime_number)