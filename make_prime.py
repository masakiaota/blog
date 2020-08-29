def ret_eratos(N: int):
    '''エラトステネスの篩'''
    is_prime = [True] * (N + 1)
    is_prime[0] = False  # 0と1は素数ではない
    is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if i % 100 == 0:
            print(i)
        if is_prime[i]:
            for j in range(i * 2, N + 1, i):  # iの倍数は素数でない
                is_prime[j] = False
    return is_prime


def _make_prime_numbers(N: int):
    # N以下の素数を列挙 -> set
    is_prime = ret_eratos(N)

    primes = set()
    for i, flg in enumerate(is_prime):
        if i % 10000000 == 0:
            print(i)
        if flg:
            primes.add(i)

    return primes


with open('primes_10^6.txt', 'w') as f:
    res = _make_prime_numbers(10**6)
    print(res, file=f)
print(len(res))
