import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class neural_net:
  def __init__(self, iters, N, m, value, output_actvn):
    self.m = m
    self.X, self.Y = self.generate_data(1, N)
    self.train_test()
    #self.view_points(self.X, self.Y)
    self.define_structure(self.X_train, self.y_train)
    self.initialize_parameters(self.input_unit, self.hidden_unit, self.output_unit, value)

    W1 = self.parameters['W1']
    b1 = self.parameters['b1']
    W2 = self.parameters['W2']
    b2 = self.parameters['b2']

    for i in range(0, iters):
      A2, cache = self.forward_prop(self.X_train, self.parameters, "identity")
      #self.cross_entropy_cost(A2, self.Y, self.parameters)
      self.least_squares_error(A2, self.Y, N)
      self.back_prop(self.parameters, cache, self.X_train, self.Y)
      self.gradient_descent(self.parameters, self.grads)
      if i%1 == 0:
        print("Cost after iteration %i: %f "%(i, self.cost))
        print(self.grads.values())

    prediction = self.prediction(self.parameters, self.X_train)

  def generate_data(self, x_roof, N):
    X = np.array([[1, 1]])
    Y = [1]
    #X = []
    #Y = []
    #for idx in range(N):
    #  x1, x2 = [np.random.uniform(0, x_roof) for i in range(2)]
    #  y = np.random.randint(-1, 1)
    #  Y.append(y)
    #  x = np.array([x1, x2])
    #  X.append(x)
    return X, np.array(Y)

  def view_points(self, x, y):
    x = np.array(x)
    y = np.array(y)

    plt.scatter(x[:,0], x[:,1], alpha=0.2, c=y, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    #plt.show()

  def train_test(self):
    #self.X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
    self.X_train = np.array(self.X.T)#.train.T
    self.y_train = np.array(self.Y)#y_train.reshape(1,self.y_train.shape[0])

    self.X_test = np.array(self.X.T)#.train.T
    self.y_test = np.array(self.Y)#y_train.reshape(1,self.y_train.shape[0])

    print ('Train X Shape: ', self.X_train.shape)
    print ('Train Y Shape: ', self.y_train.shape)
    print ('I have m = %d training examples!' % (self.X_train.shape[1]))
    print ('\nTest X Shape: ', self.X_test.shape)    


  def define_structure(self, X, Y):
    self.input_unit = X.shape[0]
    self.hidden_unit = self.m
    self.output_unit = Y.shape[0]
    print("The size of the input layer is:  = " + str(self.input_unit))
    print("The size of the hidden layer is:  = " + str(self.hidden_unit))
    print("The size of the output layer is:  = " + str(self.output_unit))    

  
  def initialize_parameters(self, input_unit, hidden_unit, output_unit, value):
    w1 = np.full((hidden_unit, input_unit), value)
    b1 = np.zeros((hidden_unit,1))
    w2 = np.full((output_unit, hidden_unit), value)
    b2 = np.zeros((output_unit,1))
    self.parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}


  def sigmoid(self, z):
    return 1/(1+ np.exp(-z))

  def forward_prop(self, X, params, output_actvn=None):
    #print(params.values())
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    if (output_actvn == "identity"):
      A2 = Z2
    elif (output_actvn == "tanh"):
      A2 = tanh(Z2)
    else:
      A2 = self.sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1,"Z2": Z2, "A2": A2}
    return A2, cache


  def cross_entropy_cost(self, A2, Y, parameters):
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = -np.sum(logprobs)/self.m
    self.cost = float(np.squeeze(cost))


  def least_squares_error(self, A2, Y, N):
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


  def prediction(self, parameters, X):
    A2, cache = self.forward_prop(X, parameters)
    predictions = np.round(A2)
    return predictions

if __name__ == "__main__":
  num_pts = 1
  m = 2 #number of hidden units in 1
  iters = 100  # number of iterations
  neural_net(iters, num_pts, m, 0.25, None)




















