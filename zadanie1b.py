#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import Ridge


class RidgeRegr:
    def __init__(self, alpha = 0.0):
        self.alpha = alpha

    def fit(self, X, Y):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta (w przyblizeniu) minimalizujace kwadratowa funkcje kosztu L uzywajac metody iteracyjnej.
        # trzeba napisać jakąś pętlę która będzie robić:
        # Q(n + 1) =  Q(n) - c*gradient
        #wystartuje od Q = 
        c = 0.0005 #na ten moment niech będzie takie
        
        # L(y, yhat) y - wart. prawdziwa/ yhat - wartość przewidziana przez model
        #yhat[i] = Theta[0] + Theta[1]*x[i][1] + ... + Theta[m]*x[i][m]
        # = sum(i=1,n)((y[i] - yhat[i])^2) + alpha*sum(i=1,m)(theta[i]^2)

        n = X.shape[0]
        k = Y.shape[0]
        ones = np.ones((n,1))
        X = np.concatenate((ones,X),axis = 1)
        m = X.shape[1]

        self.theta = np.array([0.5 for i in range(m)])
        learn = 0
        while learn < 100000: #odpalamy pętlą uczącą
            #print(f"loop {learn}")
            #yhat to tabela z pomnożonego X@Theta
            yhat = np.matmul(X, self.theta) #daje nam macierz potrzebną do 
            diff = (Y - yhat)
            
            diff = (-2)*np.transpose(diff)
            
            thet = self.theta.copy()
            thet[0] = 0
            thet = (2*self.alpha)*thet

            
            grad = np.matmul(diff,X) + thet
            self.theta = self.theta - c*grad

            learn += 1
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (k, m)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, m = X.shape
        ones = np.ones((k,1))
        X = np.concatenate((ones,X),axis = 1)
        return np.matmul(X,self.theta)


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    


test_RidgeRegressionInOneDim()

test_RidgeRegressionInThreeDim()
