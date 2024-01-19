import random

a = ['a', 'b', 'c','d','e']
b = [1, 2, 3, 4, 5]

c = list(zip(a, b))

random.shuffle(c)

a, b = zip(*c)

print (a)
print(b)
