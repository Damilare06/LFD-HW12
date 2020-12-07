import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


class neural_net:
  def __init__(self, dataset, epochs, m, value, output_actvn):
    self.m = m
    batch_size = 10
    loss_history = []
    self.W1 = self.W2 = self.b1 = self.b2 = 0
    self.vW1 = self.vW2 = self.vb1 = self.vb2 = 0

    self.split_data(dataset)
    self.view_shapes()
    self.define_structure(self.X_train, self.y_train)
    self.initialize_parameters(self.input_unit, self.hidden_unit, self.output_unit, value)

    for epoch in range(0, epochs):
      E_in = []
      self.dW1 = self.dW2 = self.db1 = self.db2 = 0

      for (X_batch, y_batch) in self.next_batch(self.X_train, self.y_train, batch_size):
        A2, cache = self.forward_prop(X_batch, self.parameters, False, output_actvn)
        self.least_squares_error(A2, X_batch, y_batch)
        self.back_prop(self.parameters, cache, X_batch, y_batch)
        self.adapt_gradient(self.grads)
        E_in.append(self.cost)

        if epoch%100000 == 0:
          print("Cost after iteration %i: %f "%(epoch, self.cost))

      self.adapt_descent()
      loss_history.append(np.average(E_in)) 
      #for k, v in self.grads.items():
      #      print( k, v)
    prediction = self.prediction(self.parameters, self.X_train)
    self.plot_decision_boundary(epochs, loss_history)

  def next_batch(self, X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
      yield(X[i:i+batch_size], Y[i:i + batch_size])

  def view_shapes(self):
    print ('Train dataset Shape: ', self.train_data.shape)    
    print ('Train X Shape: ', self.X_train.shape)
    print ('Train y Shape: ', self.y_train.shape)
    print ('I have m = %d training examples!' % (self.train_data.shape[0]))


  def define_structure(self, X, Y):
    self.input_unit = X.shape[0]
    self.hidden_unit = self.m
    self.output_unit = Y.shape[0]
    print("The size of the input layer is:  = " + str(self.input_unit))
    print("The size of the hidden layer is:  = " + str(self.hidden_unit))
    print("The size of the output layer is:  = " + str(self.output_unit))    

  
  def initialize_parameters(self, input_unit, hidden_unit, output_unit, value):
    w1 = np.random.randn(hidden_unit, input_unit)* 0.01
    w2 = np.random.randn(output_unit, hidden_unit)* 0.01
    b1 = np.zeros((hidden_unit, 1))
    b2 = np.zeros((output_unit, 1))
    self.parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}


  def sigmoid(self, z):
    return 1/(1+ np.exp(-z))

  def forward_prop(self, X, params, pred, output_actvn=None):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    if (output_actvn == "linear"):
      A2 = Z2
    elif (output_actvn == "tanh"):
      A2 = np.tanh(Z2)
    else:
      A2 = self.sigmoid(Z2)
    if (pred):
      A2 = np.sign(Z2)

    cache = {"Z1": Z1, "A1": A1,"Z2": Z2, "A2": A2}
    return A2, cache


  def cross_entropy_cost(self, A2, Y, parameters):
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = -np.sum(logprobs)/self.m
    self.cost = float(np.squeeze(cost))


  def least_squares_error(self, A2, X, Y):
    N = X.shape[1]
    error = np.sum((A2 - Y)**2)
    error  = error/(4 * N)
    self.cost = float(np.squeeze(error))
    

  def back_prop(self, params, cache, X, Y):
    m = self.m
    W1 = params['W1']
    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2,"db2": db2}
    
  def gradient_descent(self, params, grads, lr = 0.01):
    #TODO: switch this to variable gradient descent
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    
    self.parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}

  def adapt_gradient(self, grads):
    self.dW1 += grads['dW1']
    self.db1 += grads['db1']
    self.dW2 += grads['dW2']
    self.db2 += grads['db2']
    

  def adapt_descent(self, lr = 0.01, eps = 1e-8):
    self.vW1 += self.dW1**2
    self.vb1 += self.db1**2
    self.vW2 += self.dW2**2
    self.vb2 += self.db2**2
    
    self.W1 -= (lr / np.sqrt(self.vW1) + eps) * self.dW1
    self.b1 -= (lr / np.sqrt(self.vb1) + eps) * self.db1
    self.W2 -= (lr / np.sqrt(self.vW2) + eps) * self.dW2
    self.b2 -= (lr / np.sqrt(self.vb2) + eps) * self.db2
    
    self.parameters = {"W1": self.W1, "b1": self.b1,"W2": self.W2,"b2": self.b2}



  def prediction(self, parameters, X):
    A2, cache = self.forward_prop(X, parameters, True)
    predictions = np.round(A2)
    return predictions

#####################################################################################
# Input from HW9

  def split_data(self, data_set):
      index = np.array(random.sample(range(len(data_set)), 300))
      test_index = np.delete(np.arange(len(data_set)), index)
      self.train_data = data_set[index]
      self.test_data = data_set[test_index]

      self.X_train = np.array(self.train_data[:,1:]).T
      self.X_test = np.array(self.test_data[:,1:]).T

      y_train = np.array(self.train_data[:,0])
      self.y_train = y_train.reshape(1, y_train.shape[0])
      y_test = np.array(self.test_data[:,0])
      self.y_test = y_test.reshape(1, y_test.shape[0])

  def plot_decision_boundary(self, xmax, Y_data):
    ymax = int(np.max(Y_data))
    Y = np.linspace(0, ymax, ymax+1)
    X = np.linspace(0, xmax, xmax)
    plt.plot(X, Y_data)
    plt.title('Neural Networks Plot')
    plt.xlabel('iterations')
    plt.ylabel('E_in')
    plt.show()


def normalize(data_set):
    # transfer feature one by one
    data_set = data_set.astype(np.float32)
    for i in range(1, len(data_set[0])):
        max1 = np.max(data_set[:,i])
        min1 = np.min(data_set[:,i])
        diff = np.max(data_set[:,i]) + np.min(data_set[:,i])
        data_set[:,i] = 1.0*(data_set[:,i] - min1 - (max1-min1)/2) / ((max1-min1)/2)
    return data_set


def data_process(files):
    digit1, not_digit1 = [], []
    for file in files:
        raw_data = np.loadtxt(file)
        data = raw_data[:, 1:]
        # target_vector = raw_data[index, 0]
        for index in range(len(data)):
            # print(index)
            number = data[index].reshape((16, 16))
            # cv2.imshow("test", number)
            # cv2.waitKey(0)
            # feature 1. whether vertical symmetric
            number_flip = cv2.flip(number, 0)
            # more count means more unsymmetrical
            count = len(np.where(number != number_flip)[0])
            # number_final = number_flip - number

            # this is feature for pixel range
            # pixel_index = np.where(number > -1.0)[1]
            # pixel_range = max(pixel_index) - min(pixel_index)

            intensity = len(np.where(number > -1.0)[0])

            digit1.append([int(raw_data[index, 0]) == 1, count, intensity])
            # else:
            #     digit1.append((0, count, pixel_range))
    # not_digit1 = np.array(not_digit1)
    digit1 = np.array(digit1)
    return digit1 # , not_digit





if __name__ == "__main__":
  train_file = "ZipDigits.train"
  test_file = "ZipDigits.test"

  data_set = data_process([train_file,test_file])
  data_set = normalize(data_set)
  data_set[np.where(data_set[:,0]==1 ),0 ] = 1
  data_set[np.where(data_set[:,0]==0 ),0 ] = -1

  m = 10 #number of hidden units in 1
  epochs = 2000000  # number of iterations

  # Y values are the first index of the dataset 
  neural_net(data_set, epochs, m, 0.25, "linear")



















