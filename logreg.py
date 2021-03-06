import numpy
import theano
import theano.tensor as T

rng = numpy.random

N = 400        # training sample size
feats = 784    # number of input variables

# generate a random dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size = N, low = 0, high = 2))
training_steps = 10000

# declare theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name = 'w')
b = theano.shared(0., name = 'b')

print("Initial model: ")
print(w.get_value())
print(b.get_value())

# construct theano expression graph
p1 = 1 / (1 + T.exp(-T.dot(x, w) - b))                 # probability that target = 1
prediction = p1 > 0.5                                  # prediction threshold
xent = -y * T.log(p1) - (1 - y) * T.log(1 - p1)        # cross-entropy loss function (controls learning of the weights)
cost = xent.mean() + 0.01 * (w ** 2).sum()             # cost to minimize
gw, gb = T.grad(cost, [w, b])                          # compute the gradient of the cost
                                                       # w.r.t weight vector w and bias term b

# compile
train = theano.function(inputs = [x, y], outputs = [prediction, xent], updates = ((w, w - 0.1 * gw), (b, b - 0.1*gb)))
predict = theano.function(inputs = [x], outputs = prediction)

# train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])                    # D[1] is the target values
print("prediction on D:")
print(predict(D[0]))           # D[0] is the input values
