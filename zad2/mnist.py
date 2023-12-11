#!/usr/bin/env python3

import numpy as np
from neuro import Model
from loader import load_data, vectorized_result


train, validation, test = load_data()


train_wektors = [np.array(train[0][i]).reshape((-1,1)) for i in range(len(train[0]))]

Y = [np.array(vectorized_result(train[1][i])).reshape((-1,1)) for i in range(len(train[0]))]
del train

test_wektors = [np.array(test[0][i]).reshape((-1,1)) for i in range(len(test[0]))] 
test_ans = test[1]
del test

web1 = Model(1)
web1.add_input(784, 150)
web1.add_layer(75)
web1.add_layer(10)

web1.activate()
print(f"{web1.test(test_wektors,test_ans)*100}%")
web1.fit(train_wektors, Y)
print(f"{web1.test(test_wektors,test_ans)*100}%")
web1.save("mnist_webv1")
