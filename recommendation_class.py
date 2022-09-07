import numpy as np

class RecSys:
    def __init__(self, name, sex, profession):
        self.name = name
        self.sex = sex
        profession = self.profession

    def show(self):
        print(f'My name is {self.name}')

prova = RecSys(name='Gianni', sex=34, profession=True)
print(prova.show())

