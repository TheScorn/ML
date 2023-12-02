#!/usr/bin/env python3

from neuro import sigmoid
from neuro import derivsig
from neuro import Model
import numpy as np

#test czy sie czegos nauczy ta sieć

def create_test(ile:int):
    punkty = []
    expected = []
    for i in range(ile):
        punkt = np.random.randn()
        punkty.append(np.array([punkt]))
        if punkt >= 0: expected.append(np.array([[1],[0]]))
        else: expected.append(np.array([[0],[1]]))
    return(punkty,expected)

    


if __name__ == "__main__":
    punkty, exp = (create_test(50000))
    web0 = Model(0)
    web0.add_input(1,3)
    web0.add_layer(2)
    print(web0)
    web0.activate()
    print(web0)
    web0.fit(punkty, exp,learning_rate = 0.001)
    print(web0.predict(np.array([-0.02])))
    print(web0.predict(np.array([0.03])))
    print(web0.predict(np.array([-0.01])))
