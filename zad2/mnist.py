#!/usr/bin/env python3

import numpy as np
from neuro import Model
from loader import load_data, vectorized_result


def mnistlearn():
    train, validation, test = load_data()


    train_wektors = [np.array(train[0][i]).reshape((-1,1)) for i in range(len(train[0]))]

    Y = [np.array(vectorized_result(train[1][i])).reshape((-1,1)) for i in range(len(train[0]))]
    del train

    test_wektors = [np.array(test[0][i]).reshape((-1,1)) for i in range(len(test[0]))] 
    test_ans = test[1]
    del test

    #web0.add_input(784,256)
    #web0.add_layer(64)
    #web0.add_layer(10)
    #web0.activate()

    web0 = Model.load("pokazowa2")


    print(f"{web0.test(test_wektors,test_ans)*100}%")
    web0.fit(train_wektors, Y, test_wektors, test_ans, learning_rate = 0.1)
    print(f"{web0.test(test_wektors,test_ans)*100}%")
    web0.save("pokazowa3")

def mnistlearn2():
    pass

def test():
    web1 = Model.load("")
    print(web1)

if __name__ == "__main__":
    #test()
    mnistlearn()
