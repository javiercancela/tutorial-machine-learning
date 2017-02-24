# pylint: disable=I0011,C0111,C0103,E1101
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = None
        self.logfile = None

    def fit(self, X, y):
        # Creamos un array inicializado a cero. El número de elementos será
        # el número de elementos de la segunda dimensión de X, más 1
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # Iteramos tantas veces como indique n_iter
        for i in range(self.n_iter):
            print("Iteración " + str(i))
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def array_to_file(name, data, logfile, desc=''):
    logfile.write('\nVariable: {0}\n'.format(name))
    logfile.write('\Descripción: {0}\n'.format(desc))
    logfile.write('# Array shape: {0}\n'.format(data.shape))
    logfile.write(str(data))

def print_header(header_text, logfile):
    logfile.write('\n\n\n' + 50 * '=' + '\n')
    logfile.write(header_text + '\n')
    logfile.write(50 * '-' + '\n')

def open_data(logfile):
    print_header('Section: Training a perceptron model on the Iris dataset', logfile)
    df = pd.read_csv(r'.\data\iris.data', header=None)
    logfile.write(df.tail().to_string())
    return df

def plot_data(df, logfile):
    print_header('Plotting the Iris data', logfile)

    # Para los 100 primeros valores cogemos la columna 4, que indica
    # si es Iris-setosa o Iris-virginica
    y = df.iloc[0:100, 4].values
    # Sustituimos Iris-setosa por -1, e Iris-virginica por 1
    y = np.where(y == 'Iris-setosa', -1, 1)
    array_to_file('y', y, logfile)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values
    array_to_file('X', X, logfile)

    # # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.show()

def train_model(ppn):
    logfile = ppn.logfile
    print_header('Training the perceptron model', logfile)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

    plt.show()

ppn_global = Perceptron(eta=0.1, n_iter=10)

with open(r'./logs/logfile.txt', 'w') as ppn_global.logfile:
    df_global = open_data(ppn_global.logfile)
    plot_data(df_global, ppn_global.logfile)
    train_model(ppn_global)


