from typing import List
from math import *


def hello(name = None) -> str:
    if name is None or name is '':
        return "Hello!"
    else:
        return f"Hello, {name}!"


def int_to_roman(num) -> str:
    res = ''
    numerals = [[1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1],['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X','IX', 'V', 'IV', 'I']]
    for i in range(len(numerals[1])):
        while num >= numerals[0][i]:
            res += numerals[1][i]
            num -= numerals[0][i]
    return res


class BankCard():
    def __init__(self, total_sum_inp, balance_limit_inp=-1):
        self.total_sum = total_sum_inp
        self.balance_limit = balance_limit_inp
    def __call__(self, sum_spent):
        if self.total_sum - sum_spent < 0:
            print(f"Not enough money to spend {sum_spent} dollars.")
            raise ValueError
        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")
    def __str__(self):
        return "To learn the balance call balance."
    def __getattr__(self, name):
        if name == "balance":
            if self.balance_limit > 0:
                self.balance_limit -= 1
            elif self.balance_limit == 0:
                print("Balance check limits exceeded.")
                raise ValueError
            return self.total_sum
    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, b):  # a + x
        d = BankCard(0,0)
        d.total_sum = self.total_sum + b.total_sum
        if (self.balance_limit == -1) or (b.balance_limit == -1):
            d.balance_limit = -1
        else:
            d.balance_limit = max(self.balance_limit, b.balance_limit)
        return d

def primes():
    i = 1
    while True:
        i += 1
        if i == 2:
            yield i
        else:
            for j in primes():
                if i % j == 0:
                    break
                elif j > sqrt(i):
                    yield i
                    break

def longest_common_prefix(strs_input: List[str]) -> str:

    if len(strs_input) == 0:
        return  ''

    refstr = strs_input[0].lstrip()
    reslen = len(refstr)

    for i in range (len(strs_input)):
        strs_input[i] = strs_input[i].lstrip()
        reslen = min(reslen, len(strs_input[i]))


    for i in range(len(strs_input)):
        for j in range(reslen):
            if (strs_input[i])[j] != refstr[j]:
                reslen = j
                break
    return refstr[:reslen]
