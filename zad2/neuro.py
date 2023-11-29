#!/usr/bin/env python3

import numpy as np
from loader import load_data
import pickle

def sigmoid(x:float) -> float:
    return(1/(1 + np.exp(-x)))

class Web:
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
            ret = f"Web:\n"
            for i in range(len((self.matryce))):
                ret = ret + f"{self.matryce[i]}\n -> \n"
            return(ret)

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
            self.matryce = [(np.random.randn(self.input[1],self.input[0]))/np.sqrt(self.input[1])]

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
            
        if mem: 
            memory = []
            layer_memory = [None,None]
        for i in range(self.matrices - 1):
            wektor = np.matmul(self.matryce[i],wektor)
            if mem: layer_memory[0] = wektor
            wektor = sigmoid(wektor)
            if mem: 
                layer_memory[1] = wektor
                memory.append(layer_memory)
            wektor = np.vstack([np.array([1]),wektor])
            
            
        wektor = sigmoid(np.matmul(self.matryce[-1],wektor))
        
        if mem: return(wektor, memory)
        else: return(wektor)

    def predict(self,wektor):
        wektor = self.__fwd_prop(wektor)
        return(np.argmax(wektor))

#najtrudniejsze na koniec
#dane to idk co ma być na razie
    def fit(self, dane, epochs:int):
        pass

#puszcza raz fwd z memory i zapisuje wyniki
#potem na podstawie tego puszcza bwd i modyfikuje

    def __bwd_prop(self,wektor,memory):
        pass

def main():
    web1 = Web(0)
    web1.add_input(3, 2)
    web1.add_layer(2)
    print(web1)
    web1.activate()
    print(web1)
    
    wektor = np.array([[1],[1],[1]])
    print(web1.predict(wektor))

if __name__ == "__main__":
    main()
