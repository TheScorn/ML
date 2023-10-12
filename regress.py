#!/usr/bin/env python3
import numpy as np
#funkcja kosztu
#próbujemy ją minimalizować
#L(y,yhat) = sum(1 do n)((yi - yhati)^2)

#metoda iteracyjna
#wyznaczamy kolejne przybliżenia
#potrzebujemy pochodnej
#gdy jest dodatnia to idziemy w lewo 
#gdy ujemna idziemy w prawo

#1 wybieramy theta0 -> początkowe
#2 theta(n+1) = theta(n) - c*gradient(L) po theta
#c to krok uczenia

#gradient
#

def main():
    a = np.matrix([[1,2],[3,4]])
    b = np.matrix([[1],[1]])
    print(b)

    new = np.concatenate((b,a),axis=1) #przyda się później
    print(new)

    X = np.arange(15).reshape((5,3))
    print(X)

    #np.linalg.det(A) #wyznacznik
    #np.transpose(A) #transpozycja
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    n, m = X.shape
    print(np.ones((n,1)))






if __name__ == "__main__":
    main()
    import doctest
    #main()
