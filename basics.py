import theano
import theano.tensor as T


# Computing the logistic function two different ways
print("Computing the logistic function two different ways:")
x = T.dmatrix('x')
s1 = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s1)
print(logistic([[0, 1], [-1, -2]]))

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)
print(logistic2([[0, 1], [-1, -2]]))
print("")


# Computing multiple things at the same time
print("Computing multiple things at the same time:")
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f1 = theano.function([a, b], [diff, abs_diff, diff_squared])
print(f1([[1, 1], [1, 1]],[[0, 1], [2, 3]]))
print("")


# Setting default value for an argument
print("Setting default value for an argument:")
x, y = T.dscalars('x', 'y')
z = x + y
f2 = theano.function([x, theano.In(y, value = 1)], z)
print(f2(33))
print(f2(33, 2))
print("")


# Using shared parameter for functions
print("Using shared parameter for functions:")
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state + inc)])
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(300)
print(state.get_value())
state.set_value(-1)
accumulator(1)
print(state.get_value())
state.set_value(0)
decrementor = theano.function([inc], state, updates = [(state, state - inc)])
decrementor(2)
print(state.get_value())
print("")


# Using givens parameter for functions
print("Using givens parameter for functions:")
state.set_value(0)
fn_of_state = state * 2 + inc
foo = T.scalar(dtype = state.dtype)
skip_shared = theano.function([inc, foo], fn_of_state, givens = [(state, foo)])
print(skip_shared(1, 3))
print(state.get_value())
print("")


# Copying functions (starting with accumulator from above)
print("Copying functions (starting with accumulator from above):")
new_state = theano.shared(10)
new_accumulator = accumulator.copy(swap = {state : new_state})
new_accumulator(100)
print(state.get_value())
print(new_state.get_value())
null_accumulator = accumulator.copy(delete_updates = True)
null_accumulator(9000)
print(state.get_value())
print("")









