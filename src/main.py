import numpy as np
from nn import General

"""
class a :
    def __init__(self):
        self.sayi = np.random.uniform(-0.1, 0.1)

class b : 
    def __init__(self, sayi):
        self.sayilar = [a() for i in range(sayi)]

class c :
    def __init__(self, sayi):
        self.bs = list()

        for i in range(sayi):
            self.bs.append(b(5))
    
    def my_print(self):
        print(self.bs[0].sayilar[0].sayi)
"""    

if __name__ == "__main__":
    inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    genel = General([12, 8, 4, 2], inputs)
    genel.forward()