def multiply(x):
    return x*10

numbers = [1,2,3,4]

result = map(multiply,numbers)

print(result)   #<map object at 0x00000167AE1FACB0>

print(list(result)) #[10, 20, 30, 40]