import numpy as np
import matplotlib.pyplot as plt

# Generamos los datos
# Área (m²) y Precios (en miles de USD)
X = np.array([50, 60, 70, 80, 90, 100, 110, 120])       # Área
y = np.array([150, 180, 200, 240, 260, 300, 320, 350])  # Precio
print("x:", X)
print("y:", y)

# Reshaping (to one column) para hacer compatible con el modelo
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
print("x reshape:\n", X)
print("y reshape:\n", y)

# Añadimos un 1 al frente de X (other column) para incluir el término independiente (theta_0)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
print("theta0 + x:\n", X_b)

# Inicializamos los parámetros theta (theta_0, theta_1)
thetas = np.random.randn(2, 1) # Matrix 2x1 = [[theta_0], [theta_1]]
print("thetas:\n", thetas)

# Definimos la tasa de aprendizaje y el número de iteraciones
alpha = 0.0001
iterations = 1000
m = len(X_b)

predictions = X_b.dot(thetas) # If X_b=8x2 and thetas=2x1, so pred=8x1.
print("X_b dot thetas:\n", predictions)
print("Cost:", (1/(2*m)) * np.sum((predictions - y)**2))


# Función de Costo (MSE: median square error): measures the error between the model's predictions and the actual data values.
def compute_cost(thetas, X_b, y):
    # Prediction model = Linear regression (in this case the simplest linear model for regression): if X_b=8x2 and thetas=2x1, so pred=8x1.
    predictions = X_b.dot(thetas)
    cost = (1/(2*m)) * np.sum((predictions - y)**2) # MSE
    return cost

# Descenso de Gradiente (to minimize cost function to get the best model): the objective of training is to minimize the cost function.
def gradient_descent(X_b, y, thetas, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        # gradients = Derivate of cost function (MSE): 
            # La notación XT(Xθ−y) es simplemente una forma compacta y eficiente de expresar lo que antes se hacía sumando a mano para cada θj​.
            # La transposición de X se usa para asegurarse de que el resultado de la multiplicación XT(Xθ−y) tenga la forma correcta de un vector de gradientes de tamaño n×1, adecuado para actualizar los parámetros θ.
        gradients = (1/m) * X_b.T.dot(X_b.dot(thetas) - y) # X_b.T = transpose
        thetas = thetas - alpha * gradients
        #print("It ", i, ", thetas:\n", thetas)
        cost_history[i] = compute_cost(thetas, X_b, y)
        #print("It ", i, ", cost: ", cost_history[i])
    return thetas, cost_history

# Ejecutamos el descenso de gradiente
thetas_optimal, cost_history = gradient_descent(X_b, y, thetas, alpha, iterations)
model_found = X_b.dot(thetas_optimal)

# Mostramos los resultados
print(f"Valor final de la función de costo: {cost_history[-1]}")
print(f"Parámetros óptimos:\n {thetas_optimal}")

# Graficamos los resultado
plt.plot(X, y, label='Datos reales', color='red')
plt.plot(X, model_found, label='Modelo ajustado', color='blue')

plt.xlabel('Área (m²)')
plt.ylabel('Precio (en miles de USD)')
plt.legend()
plt.title('Regresión Lineal para Predecir Precio de Casas')
plt.show()

