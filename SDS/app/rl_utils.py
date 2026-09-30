import numpy as np

def discretize(state):
    return tuple(int(x) for x in state)

def q_learning_predict(state, Q):
    state = discretize(state)
    return Q.get(state, np.zeros(2)).argmax()
