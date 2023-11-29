notatki z 25 z wykładu


Rozważmy gęstą sieć neuronową o L warstwach.
Powiedzmy, że wejście jest wektorem a^[0] e R^(n^([0])), a wyjście
wektorem a^[L] e R^(n^([L])).

W wartstwie l e {1,2,...,L} mamy macierz wag W^[L`]

wymiaru n^[l] x (n^[l-1]+1), W^[l] = [w(j,k)^[l]] j = 1,...,n^[l]
                                                  k = 0,...,n^[l-1]


-Przejście w przód, czyli obliczanie wartości (forward pass)

Załóżmy, że mamy już waktor a^[l], gdzie l e {0,1,...,L-1}. Tworzymy wektor
x^[l] dołączając do a[l] jedynkę z przodu:

x^[l] = [ 1 a1^[l] a2^[l] ... an^[l]^[l] ] (<-tylko że w pionie) e R^(n^([l])+1)
Liczymy net^[l+1] = W^[l+1] * x^[l] e R.
Ostatecznie kłądziemy  a^[l+1] = phi(net^[l+1]) = 
= [ phi(net1^[l+1]) phi(net2^[l+1]) ... phi(netn^[l+1]) ]


W^[e] = [Wj,k] j = 1,...,n^[l] k=0,1,...,n^[l-1]
net^[l+1] = W^[l+1] * x^[l]
                            phi(net1^[l+1])    
a^[l+1] = phi(net^[l+1]) = [     ...       ] = 
                            phi(netn^[l+1])

= [phi(sum(j = 0;n^[l]) W^[l+1]k,j * x^[l]j)]k=1,2,...,n^[l+1]


-Propagacja wstecz (backward pass/propagation). Powiedzmy, że oczekiwaliśmy
wyjścia równego y e R^(n^[l])

Wprowadzamy funkcję kosztu L(a^[l],y) = 
= (1/2)*sum{k=1;n^[l]}(a^[l]k - yk)^2   
(naszym celem jest obliczenie dL/dW^[l]k,j)

Najpierw obliczymy dL/da^[l]k = a^[l]k - y.
Powiedzmy, że mamy już dL/da^[l]k (gdzie l e {1,2,...,L}).
Ustalmy j0 e {1,...,n^[l]}, j0 e {0,1,...,n^[l-1]}

a^[l] = [phi(sum{j=0;n^[l-1]}(W^[l]k,j * x^[l-1]j))]k = 1,2,...,n^[l]


(Różniczka wielu zmiennych:
phi(ak)    phi'(ak)
(d/dw) phi(a1(w),a2(w)) = dphi/da1(...)*a1'(w) + dphi/da2(...)*a2'(w)
)

Czyli liczymy 
dL/dW^[l]k0,j0  = sum{k=1;n^[l]}((dL/d^a^[l]k)*(da^[l]k/dW^[l]k0,j0)) = 
                                                             ^
                                                             0 dla k!=k0

= dL/d^[l]k0 * phi'(sum{j=0;n^[l-1]}(W^[l]k0,j0 * x^[l-1]j)) * x^[l-1]j0 =:

=: del^[l]k0 * x^[l-1]j0

                                                 del^[l]1
dL/dW^[l] = ?macierz? = del^[l] * (x^[l-1])^T = [   ...  ] @ [x0 ... xn] =
                                                 del^[l]n


(trzeba zapamiętywać net przy przejściu w przód by potem szybciej przejść w tył)


-Liczymy dL/da^[l-1]k0 = dL/dx^[l-1]k0 =

= sum{k=1;n^[l]}(dL/d)


