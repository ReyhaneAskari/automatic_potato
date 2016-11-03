import theano
import theano.tensor as T
import numpy as np
import time
 
start_time = time.time()
X = T.fmatrix('X')

hidden_layers_list = [X]
weights_list = []
layers_size = [1000, 500, 250, 250, 250, 250, 250, 250, 250, 10]

for i in range(len(layers_size) - 1):
    w = theano.shared(np.random.rand(
        layers_size[i], layers_size[i + 1]).astype('float32'), 'w' + str(i))
    weights_list.append(w)
    h = T.dot(hidden_layers_list[i], w)
    hidden_layers_list.append(h)

f = theano.function([X], hidden_layers_list)

print f(np.random.rand(100, 1000).astype('float32'))[-1].shape
print time.time() - start_time
