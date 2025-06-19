import numpy as np
import matplotlib.pyplot as plt

# 121 data cars -> [antique, cost (cost of going to market)]

# n(rows) x 2(cols)
x = np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0], [1.0, 1.0], 
              [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9], [0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9], [0.9, 0.9], [1.0, 0.9], 
              [0.0, 0.8], [0.1, 0.8], [0.2, 0.8], [0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8], [0.8, 0.8], [0.9, 0.8], [1.0, 0.8], 
              [0.0, 0.7], [0.1, 0.7], [0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7], [0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], 
              [0.0, 0.6], [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6], [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
              [0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5], [0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5], [1.0, 0.5], 
              [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4], [0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4], [0.9, 0.4], [1.0, 0.4], 
              [0.0, 0.3], [0.1, 0.3], [0.2, 0.3], [0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3], [0.8, 0.3], [0.9, 0.3], [1.0, 0.3], 
              [0.0, 0.2], [0.1, 0.2], [0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2], [0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], 
              [0.0, 0.1], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1], [0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
              [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0], [1.0, 0.0]])

# 0 : normal;  1 : collectable.
# nx1
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
              0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
              0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
              0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
              0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,])

# Dispersion graph.
plt.figure(figsize=(8, 6))
plt.title("Data training", fontsize=14)

plt.scatter(x[y == 0].T[0], x[y == 0].T[1], # x[y == 0]: This applies Boolean selection to x. It returns the rows in x whose corresponding row in y is equal to 0.
            marker="x", s=100, color="pink", linewidths=5, label="Normal")
plt.scatter(x[y == 1].T[0], x[y == 1].T[1],
            marker="o", s=100, color="purple", linewidths=5, label="Collectable")
plt.xlabel("Antique", fontsize=10)
plt.ylabel("Cost", fontsize=10)
plt.legend(bbox_to_anchor=(1.3, 0.15))
plt.box(False)
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()


# For reproducibility.
np.random.seed(0)

class RedNeuronal:
    def __init__(self, x, y):
        # Training data.
        self.x = x
        # Class associated with the training data.
        self.y = y
        self.N = len(self.y)

        # Network structure and random initialization of weights.
        self.weights1 = np.random.rand(4) # [w1,w2,w3,w4].
        self.bias1 = np.random.rand(2)    # [w00, w01]
        self.weights2 = np.random.rand(2)
        self.bias2 = np.random.rand(1)

    def fit(self, learning_rate=0.1, epoch=1000):
        # For k training epoch.
        for k in range(epoch):
            error = 0
            # For each k epoch:
            # 1) Do feed forward with each instance i.
            # 2) Calculates the squared error and gradients.
            # 3) Update weights.
            for i in range(self.N):
                # Entradas a las neuronas sigmoides ocultas.
                suma_o1 = self.x[i][0]*self.weights1[0] + self.x[i][1]*self.weights1[2] + self.bias1[0]
                suma_o2 = self.x[i][0]*self.weights1[1] + self.x[i][1]*self.weights1[3] + self.bias1[1]
                # Salidas de las neuronas sigmoides ocultas
                salida_o1 =  1/(1 + np.exp(-suma_o1))
                salida_o2 = 1/(1 + np.exp(-suma_o2))
                # Entrada de la neurona sigmoide de la capa de salida
                suma_s = salida_o1*self.weights2[0] + salida_o2*self.weights2[1] + self.bias2[0]
                # Salida de la red neuronal
                y_gorro = 1/(1 + np.exp(-suma_s))

                # Calculation of the squared error (MSE).
                error += (1/2)*(self.y[i] - y_gorro)**2

                # Backpropagation: calculation of gradients.
                gradiente_p21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o1
                gradiente_p22 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o2
                gradiente_sesgo21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * 1

                gradiente_p11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                self.weights2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][0]
                gradiente_p13 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                self.weights2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][1]
                gradiente_sesgo11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                    self.weights2[0] * (salida_o1 * (1 - salida_o1)) * 1

                gradiente_p12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                self.weights2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][0]
                gradiente_p14 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                self.weights2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][1]
                gradiente_sesgo12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
                                    self.weights2[1] * (salida_o2 * (1 - salida_o2)) * 1
                
                # Weight update (gradient descent).
                self.weights1[0] -= learning_rate * gradiente_p11
                self.weights1[1] -= learning_rate * gradiente_p12
                self.weights1[2] -= learning_rate * gradiente_p13
                self.weights1[3] -= learning_rate * gradiente_p14
                self.bias1[0] -= learning_rate * gradiente_sesgo11
                self.bias1[1] -= learning_rate * gradiente_sesgo12
                self.weights2[0] -= learning_rate * gradiente_p21
                self.weights2[1] -= learning_rate * gradiente_p22
                self.bias2[0] -= learning_rate * gradiente_sesgo21
            print(error)

    def clasificacion(self, x1, x2):
        # Forward propagation with the new instance (x1, x2) to be evaluated.
        suma_o1 = x1*self.weights1[0] + x2*self.weights1[2] + self.bias1[0]
        suma_o2 = x1*self.weights1[1] + x2*self.weights1[3] + self.bias1[1]
        salida_o1 = 1/(1 + np.exp(-suma_o1))
        salida_o2 = 1/(1 + np.exp(-suma_o2))
        suma_s = salida_o1*self.weights2[0] + salida_o2*self.weights2[1] + self.bias2[0]
        y_gorro = 1/(1 + np.exp(-suma_s))
        return round(y_gorro)

    
# Create artificial Neuronal Network.
red_neuronal = RedNeuronal(x, y)
red_neuronal.fit()


# Explore the results of the trained Neural Network
plt.figure(figsize=(8, 6))
plt.title("Classification Results", fontsize=14)

plt.scatter(x[y == 0].T[0],
            x[y == 0].T[1],
            marker="x", s=100, color="pink",
            linewidths=5, label="Normal")
plt.scatter(x[y == 1].T[0],
            x[y == 1].T[1],
            marker="o", s=100, color="purple",
            linewidths=5, label="Collectable")

for antiguedad in np.arange(0, 1.01, 0.025):
  for costo in np.arange(0, 1.01, 0.025):
    color = red_neuronal.clasificacion(antiguedad, costo)
    if color == 1:
      plt.scatter(antiguedad, costo, marker="s", s=110,
                  color="purple", alpha=0.2, linewidths=0)
    else:
      plt.scatter(antiguedad, costo, marker="s", s=110,
                  color="pink", alpha=0.2, linewidths=0)

plt.xlabel("Antique", fontsize=10)
plt.ylabel("Cost", fontsize=10)
plt.legend(bbox_to_anchor=(1.3, 0.15))
plt.box(False)
plt.xlim((0, 1.01))
plt.ylim((0, 1.01))
plt.show()