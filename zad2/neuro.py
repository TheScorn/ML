#!/usr/bin/env python3

import numpy as np
from loader import load_data
from functions import sigmoid, derivsig
import pickle
import os

class Model:
    """
    Simple dense web class.

    Initialise planing with:
    [name] = Web([id number])

    (Id number is not important untill saving functionality is added.)
    
    Methods:
    add_input([number of input args], [number of output args])
    adds first layer
    """
    def __init__(self, number: int) -> None:
        self.number = number
        self.active = False
        self.layers = [] #lista tupli z inputem i outputem
        self.input = None
        self.matrices = 0

    def __str__(self):
        if self.active:
            ret0 = f"Web plan:\n{self.input}"
            for i in range(len(self.layers)):
                ret0 = ret0 + f" -> {self.layers[i]}"


            ret = f"\nWeb:\n"
            for i in range(len((self.matryce))):
                ret = ret + f"{self.matryce[i]}\n -> \n"
            return(ret0 + ret)

        else:
            ret = f"Web plan:\n{self.input}"
            for i in range(len(self.layers)):
                ret = ret + f" -> {self.layers[i]}"
            return(ret)

    def add_input(self, input_args, output_args):
        if self.active:
            raise Exception("cannot perform this operation on activated web!")
        else:
            self.input = (input_args, output_args)    
    
    def add_layer(self, output: int) -> None:
        if self.active:
            raise Exception("cannot perform this operation on an activated web!")
        else:
            if self.input == None:
                raise Exception("Input layer needed first")
            elif self.layers == []:
                self.layers.append((self.input[1],output))
            else:
                self.layers.append((self.layers[-1][1],output))

    def activate(self):
        """
        method activates web
        """
        if self.active:
            raise Exception("cannot activate already active web!")
        else:
            print(f"Initiating web number {self.number}")
            self.active = True
            #tutaj w grę wchodzi numpy
            #tworzymy macierze i inicjujemy wartości jakimś rozkładem
            #trzeba pamiętać że macierze mają +1 do inputu bo dochodzi bias

            #wartości początkowe są losowane z rozkładu standardowego i dzielone przez pierw z inputu
            self.matryce = [(np.random.randn(self.input[1],self.input[0] + 1))/np.sqrt(self.input[1])]

            for i in range(len(self.layers)):
                self.matryce.append(np.random.randn(self.layers[i][1],self.layers[i][0] + 1)/np.sqrt(self.layers[i][0]))
            
            self.matrices = len(self.matryce)
    
    def __fwd_prop(self,wektor,mem:bool = False):
        """
        wektor to nupyowa 1-wymiarowa tablica
        """
        if not self.active:
            raise Exception("Can't use method on unactive web. Use web.activate() before.")
        
        if len(wektor) != self.input[0]:
            raise Exception("Input vector dim does not match 1st layers dim.")
           
        wektor = np.vstack([np.array([1]),wektor])
        
        if mem: X = [wektor]
        if mem: net = []

        for i in range(self.matrices - 1):
            
            wektor = np.matmul(self.matryce[i],wektor)
            
            if mem: net.append(wektor)
            
            wektor = sigmoid(wektor)
            wektor = np.vstack([np.array([1]),wektor])
            
            if mem: X.append(wektor)
            
        wektor = np.matmul(self.matryce[-1],wektor)   
        
        if mem: 
            net.append(wektor) 
            memory = [X, net]

        wektor = sigmoid(wektor)
        #tego nie musimy zapamiętać bo to jest wynik

        if mem: return(wektor, memory)
        else: return(wektor)


    def test(self, wektor_testow, wektor_odp):
        sum_of_right = 0
        length_wt = len(wektor_testow)
        if length_wt != len(wektor_odp):
            raise Exception("Expected values have different dim than the input vector")
        for i in range(length_wt):
            wynik = self.predict(wektor_testow[i])

            if wynik == wektor_odp[i]:
                sum_of_right += 1
        return(sum_of_right/length_wt)

    def predict(self,wektor):
        wektor = self.__fwd_prop(wektor)
        return(np.argmax(wektor))


    def fit(self, wektor, expected, epochs:int=1, learning_rate: float = 0.1):
        """
        Trzeba pamiętać że wektor to tak naprawdę wektor wektorów, tak samo expected.
        """
        if len(wektor) != len(expected):
            raise Exception("Expected values have different dim than the input vector.")

        for i in range(len(wektor)):
            wynik = self.__fwd_prop(wektor[i],mem = True)
            yhat = wynik[0]
            memory = wynik[1]
            diff = yhat - expected[i]
            self.__bwd_prop(diff,memory, learning_rate)
        #puszcza raz fwd z memory i zapisuje wyniki
        #potem na podstawie tego puszcza bwd i modyfikuje



    def fit(self, wektor, expected, test, test_expected, epochs:int=1, learning_rate: float = 0.1):
        """
        Trzeba pamiętać że wektor to tak naprawdę wektor wektorów, tak samo expected.
        """
        if len(wektor) != len(expected):
            raise Exception("Expected values have different dim than the input vector.")
        for i in range(len(wektor)):
            wynik = self.__fwd_prop(wektor[i],mem = True)
            yhat = wynik[0]
            memory = wynik[1]
            diff = yhat - expected[i]
            self.__bwd_prop(diff,memory, learning_rate)
            #if (i % 1000) == 0:
                #print(f"{self.test(test, test_expected)*100}%")
        #puszcza raz fwd z memory i zapisuje wyniki
        #potem na podstawie tego puszcza bwd i modyfikuje
        

    def __bwd_prop(self,wektor,memory,learning_rate: float):
        #wektor to różnica między oczekiwanym a przewidzianym
        
        values = wektor
        new_weights = []

        for i in range(self.matrices - 1, -1, -1):
            derivatives = derivsig(memory[1][i])
            delta_sig = values * derivatives
            derW = np.matmul(delta_sig,np.transpose(memory[0][i]))
            new = self.matryce[i] - (learning_rate * derW)
            new_weights.append(new)
            derX = np.matmul(np.transpose(self.matryce[i]),delta_sig)
            dera = np.delete(derX, 0, 0)
            values = dera
    
        new_weights.reverse()

        for i in range(len(new_weights)):
            self.matryce[i] = new_weights[i]

    def save(self,filename):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir('models')
        
        
        with open(f'./{filename}', 'wb') as file:
            pickle.dump(self, file) 


    @classmethod
    def load(cls, filename):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir('models')

        with open(f'./{filename}',"rb") as file:
            ret = pickle.load(file)
        return(ret)

def main():
    web1 = Model(0)
    web1.add_input(3,3)
    web1.add_layer(3)
    web1.add_layer(2)
    web1.activate()
    print(web1)
    web1.save("web1")
    del web1
    web1ret = Model.load("web1")
    print(web1ret)
    

if __name__ == "__main__":
    main()
