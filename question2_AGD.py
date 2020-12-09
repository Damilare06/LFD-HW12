import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


class neural_net:
  def __init__(self, dataset, epochs, m, output_actvn=None, W_lambda=0.0):
    self.m = m
    loss_history, digit1, digitNot1 = [], [], []
    self.vW1 = self.vW2 = self.vb1 = self.vb2 = 0

    self.split_data(dataset)
    self.view_shapes()
    self.define_structure(self.X_train, self.y_train)
    self.initialize_parameters(self.input_unit, self.hidden_unit, self.output_unit)

    for epoch in range(0, epochs):
      #E_in = []
      #self.dW1 = self.dW2 = self.db1 = self.db2 = 0
      A2, cache = self.forward_prop(self.X_train, self.parameters, False, output_actvn)
      self.least_squares_error(A2, self.y_train)
      self.back_prop(self.parameters, cache, self.X_train, self.y_train)
      self.gradient_descent(self.parameters, self.grads)
      #self.adapt_gradient(self.grads)
      #self.adapt_descent()
      loss_history.append(self.cost)

      if epoch%500 == 0:
          print("Cost after iteration %i: %f "%(epoch, self.cost))

    #val = self.prediction(self.parameters, np.array([[0.5, 0.5],]).T)
    #print(val)

    pred_train = self.prediction(self.parameters, self.X_train)

    self.least_squares_error(A2, self.y_train)
    xvalues = self.check_predictions(pred_train)
    digit1, digitNot1 = self.analyse_data(self.train_data)
    self.plot_decision(digit1, digitNot1, xvalues)



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

  
  def initialize_parameters(self, input_unit, hidden_unit, output_unit):
    self.W1 = np.random.randn(hidden_unit, input_unit)* 0.01
    self.W2 = np.random.randn(output_unit, hidden_unit)* 0.01
    self.b1 = np.zeros((hidden_unit, 1))
    self.b2 = np.zeros((output_unit, 1))
    self.parameters = {"W1": self.W1,
                       "b1": self.b1,
                       "W2": self.W2,
                       "b2": self.b2}


  def sigmoid(self, z):
    return 1/(1+ np.exp(-z))


  def forward_prop(self, X, params, pred, output_actvn=None):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    Z1 = np.add(np.dot(W1, X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.dot(W2, A1), b2)

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


  def least_squares_error(self, A2, Y):
    N = Y.shape[1]
    error = np.sum((A2 - Y)**2)
    error  = error/(4 * N)
    self.cost = float(np.squeeze(error))
    

  def back_prop(self, params, cache, X, Y):
    m = self.m
    W1 = params['W1']
    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    self.dZ2 = A2 - Y
    self.dW2 = (1/m) * np.dot(self.dZ2, A1.T)
    self.db2 = (1/m) * np.sum(self.dZ2, axis=1, keepdims=True)
    self.dZ1 = np.multiply(np.dot(W2.T, self.dZ2), 1 - np.power(A1, 2))
    self.dW1 = (1/m) * np.dot(self.dZ1, X.T)
    self.db1 = (1/m) * np.sum(self.dZ1, axis=1, keepdims=True)

    self.grads = {"dW1": self.dW1, "db1": self.db1, "dW2": self.dW2,"db2": self.db2}
    
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
    
    self.W1 = W1 - lr * dW1
    self.b1 = b1 - lr * db1
    self.W2 = W2 - lr * dW2
    self.b2 = b2 - lr * db2
    
    self.parameters = {"W1": self.W1, "b1": self.b1,"W2": self.W2,"b2": self.b2}

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
    #plt.show()


  def analyse_data(self, data):
    digit1, digitNot1 = [], []
    for i in range(len(data[:,0])):
      if data[i][0] == 1:
        digit1.append([data[i][1], data[i][2]])
      else:
        digitNot1.append([data[i][1], data[i][2]])
    return digit1, digitNot1
  
  
  def check_predictions(self, pred):
    x1true, x2true = [], []
    x1false, x2false = [], []

    # defnining a mesh over the entire domain
    for i in np.arange(-1, 1.02, 0.02):
      for j in np.arange(-1, 1.02, 0.02):
        val = self.prediction(self.parameters, np.array([[i, j],]).T)
        if val[0,0] == 1:
          x1true.append(i)
          x2true.append(j)
        else:
          x1false.append(i)
          x2false.append(j)
    return (x1true, x2true, x1false, x2false)
  
  
  def plot_decision(self, digit1, digitNot1, xparams):
    x1true, x2true, x1false, x2false = xparams
    plt.scatter(x1true, x2true, s = 12, label = 'h(x) == 1')
    plt.scatter(x1false, x2false, s = 12, label = 'h(x) != 1')
    print(np.array(digit1).shape)
    print(np.array(digitNot1).shape)
    plt.scatter([digit1[i][0] for i in range(len(digit1))], [digit1[i][1] for i in range(len(digit1))], s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
    plt.scatter([digitNot1[i][0] for i in range(len(digitNot1))], [digitNot1[i][1] for i in range(len(digitNot1))], s = 12, c = 'red', marker = 'x', label = 'not digit 1')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    plt.show()


def normalize(data_set):
    data_set = data_set.astype(np.float32)
    for i in range(1, len(data_set[0])):
        max1 = np.max(data_set[:,i])
        min1 = np.min(data_set[:,i])
        data_set[:,i] = 1.0*(data_set[:,i] - min1 - (max1-min1)/2) / ((max1-min1)/2)
    return data_set


def data_process(files):
    digit = []
    for file in files:
        raw_data = np.loadtxt(file)
        data = raw_data[:, 1:]
        for index in range(len(data)):
            symmetry = get_symmetry(data[index])
            intensity = get_intensity(data[index])
            digit.append([int(raw_data[index, 0]) == 1, intensity, symmetry])
    digit = np.array(digit)
    return digit

def parsearr(arr):
    newarr = np.zeros((16, 16))
    for i in range(0, 16):
        for j in range(0, 16):
            newarr[i][j] = arr[i*16+j]
    return newarr

def get_symmetry(digit):
    arr = parsearr(digit)

    symmetry = 0
    for i in range(0,256,16):
        for j in range(i, i+8):
            if digit[j] == digit[2*i+15-j]:
                symmetry += 1
    return symmetry


def get_intensity(digit):
    intensity = 0
    for i in digit:
        intensity += i
    return intensity/256




if __name__ == "__main__":
  train_file = "ZipDigits.train"
  test_file = "ZipDigits.test"

  data_set = data_process([train_file, test_file])
  data_set = normalize(data_set)

  m = 10 #number of hidden units in 1
  epochs = 2000#0000  # number of iterations

  # Y values are the first index of the dataset 
  neural_net(data_set, epochs, m)#, "linear")

######################## Question 2b ############################


#  neural_net(data_set, epochs, m, "linear", 0.01)
















