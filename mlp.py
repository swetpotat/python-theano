import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):

	def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = T.tanh)

		"""
		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dmatrix
		:param input: a symbolic tensor of shape (n_examples, n_in)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_out: int
		:param n_out: number of hidden units

		:type activation: theano.Op or function
		:param activation: non linearity to be applied in the hidden layer
		"""

		self.input = input

		# initial value for weights
		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low = -numpy.sqrt(6. / (n_in + n_out)), 
					high = numpy.sqrt(6. / (n_in + n_out)), 
					size = (n_in, n_out)
				),
				dtype = theano.config.floatX 		
			)
			# initial values should the sigmoid function be used
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value = W_values, name = 'W', borrow = True)

		# initial values for the bias
		if b is None:
			b_values = numpy.zeros((n_out), dtype = theano.config.floatX)
			b = theano.shared(value = b_values, name = 'b', borrow = True)
		
		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)

		self.params = [self.W, self.b]



class MLP(object):

	def __init__(self, rng, input, n_in, n_hidden, n_out):

		"""
		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in which the datapoints lie

		:type n_hidden: int 
		:param n_hidden: number of hidden units

		:type n_out: int
		param n_out: number of output units, the dimension of the space in which the labels lie
		""" 

		self.hiddenLayer = HiddenLayer(
			rng = rng,
			input = input,
			n_in = n_in,
			n_out = n_out,
			activation = T.tanh
		)

		# output of hidden layers becomes the input for the regression (output) layer
		self.logRegressionLayer = LogisticRegression(
			input = self.hiddenLayer.output,
			n_in = n_hidden
			n_out = n_out
		)

		# L1 norm: enforced to be small for regularisation
		self.L1 = (
			abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
		)

		# square of L2 norm: enforced to be small for regularisation
		self.L2_sqr(
			(self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
		)

		# negative log likelihood of MLP is computed in the logistic regression layer
		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)

		# computes the number of errors
		self.errors = self.logRegressionLayer.errors

		# model parameters are the sum of parameters of the two layers it is composed of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

		# keep track of model input
		self.input = input
		
		# update the parameters of the model
		gparams = [T.grad(cost, param) for param in classifier.params]
		updates = [
			(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
		]
		
		# compiling a theano function 'train_model' that returns the cost
		# but at the same time updates the parameters of the model based
		# on the rules defined in 'updates'
		train_model = theano.function(
			inputs = [index],
			outputs = cost,
			updates = updates,
			givens = {
					x: train_set_x[index * batch_size: (index + 1) * batch_size],
					y: train_set_y[index * batch_size: (index + 1) * batch_size]								
				}
		)



def test_mlp(learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000, dataset = 'mnist.pkl.gz', batch_size = 20, n_hidden = 500):

	"""
	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic gradient)

	:type L1_reg: float
	:param L1_reg: L1-norm's weight when added to the cost

	:type L2_reg: float
	:param L2_reg: L2-norm's weight when added to the cost

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: the path of the MNIST dataset file
	"""

	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
	n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
	n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size	



	######################
	# BUILD ACTUAL MODEL #
	######################
	
	print('...building the model')

	index = T.lscalar()    # index to a minibatch
	x = T.matrix('x')      # the data is presented as rasterized images
	y = T.ivector('y')     # the labels are presented as 1D vectors of ints

	rng = numpy.random.RandomState(1234)

	# construct the MLP class
	classifier = MLP(
		rng = rng,
		input = x,
		n_in = 28 * 28
		n_hidden = n_hidden
		n_out = 10
	)

	# cost function (expressed symbolically)
	cost = (
		classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + l2_reg * classifier.L2_sqr
	)

	# computes mistakes that are made by the model on a minibatch
	test_model = theano.function(
		inputs = [index],
		outputs = classifier.errors(y),
		givens = {
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]								
		}
	)	















