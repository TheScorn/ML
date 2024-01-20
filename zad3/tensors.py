#!/usr/bin/env python3

import torch
import numpy as np

#tensory są podobne do arraów z np ale mogą działać na gpu

#tworzenie tensora z danych
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#bridge numpy - można tworzyć tensory z obiektów z np
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

#Przy tworzeniu z innego tensora zapamiętuje kształt i typ danych
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n") #zapamiętało wymiar


x_rand = torch.rand_like(x_data, dtype=torch.float) #zapamięta wymiar ale nadpisze typ danych z x_data
print(f"Random Tensor: \n {x_rand} \n")



#------------------------------------------------------------------
#shape definiuje kształt tensora
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random: \n {rand_tensor} \n")
print(f"Ones: \n {ones_tensor} \n")
print(f"Zeros: \n {zeros_tensor}")



#-------------------------------------------------------------
#atrybuty tensora
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
#z defaultu jest tworzony na cpu

#--------------------------------------------------------------
#jest bardzo dużo możliwych operacji na tensorach
#większość(jak nie wszystkie) te co na np array
#można je wykonywać na gpu
#aby przenieść tensor używamy .to
#po sprawdzeniu czy jest możliwość

if torch.cuda.is_available():
    tensor = tensor.to("cuda")


#standardowe niby-numpyowe operacje
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#łączenie tensorów
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)



#operacje arytmetyczne
y1 = tensor @ tensor.T #tensor.T zwraca transpozycję
y2 = tensor.matmul(tensor.T) #mnożenia macierzowe
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)


z1 = tensor * tensor #mnożenie po elementach
z2 = tensor.mul(tensor) #to samo

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)


#jednoelementowe tensory można konwertować na zwykły typ numeryczny
#można sumować wartości z tensora
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
