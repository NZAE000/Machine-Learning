import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# House features: dimentions(m²) and n rooms.
X = np.array([[50, 1],  
              [60, 2],  
              [75, 2],  
              [80, 3],  
              [90, 3],  
              [100, 4],
              [120, 4], 
              [130, 5], 
              [150, 5], 
              [160, 6], 
              [180, 6], 
              [200, 7],
              [220, 7], 
              [250, 8], 
              [280, 8], 
              [300, 9], 
              [320, 9], 
              [350, 10],
              [400, 10]])
y = np.array([150, 180, 220, 240, 270, 310, 350, 400, 450, 500, 550, 600,
              650, 700, 750, 800, 850, 900, 950])  # Price.

print("x:", X)
print("y:", y)

# Reshaping para hacer compatible con el modelo
# X = X.reshape(-1, 2) # There is no need to reshape it into a matrix with the correct format for the model, since it already has the correct shape.
y = y.reshape(-1, 1)
print("x reshape:\n", X)
print("y reshape:\n", y)

# Se añade un 1 al frente de X (other column) para incluir el término independiente (theta_0)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
print("x0 + x1 + .. + xn:\n", X_b)

# Inicializar parámetros theta (theta_0, theta_1, theta_2)
thetas = np.random.randn(3, 1) # Matrix 3x1 = [[theta_0], [theta_1], [theta_2]]
print("thetas:\n", thetas)

# Definir la tasa de aprendizaje y el número de iteraciones
alpha = 0.00001
iterations = 1000
m = len(X_b)

predictions = X_b.dot(thetas) # If X_b=8x3 and thetas=3x1, so pred=8x1.
print("X_b dot thetas:\n", predictions)
print("Cost:", (1/(2*m)) * np.sum((predictions - y)**2))


# Función de Costo (MSE): measures the error between the model's predictions and the actual data values.
def compute_cost(thetas, X_b, y):
    # Prediction model = Linear regression (in this case the simplest linear model for regression)): if X_b=8x3 and thetas=3x1, so pred=8x1.
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

# Ejecutar el descenso de gradiente
thetas_optimal, cost_history = gradient_descent(X_b, y, thetas, alpha, iterations)
model_found = X_b.dot(thetas_optimal)

# Mostrar los resultados
print(f"Valor final de la función de costo: {cost_history[-1]}")
print(f"Parámetros óptimos:\n {thetas_optimal}")

## Graficar los resultado ################################
# Extraer las dos características
dims = X[:, 0]    # Dimensión de las casas (m²)
rooms = X[:, 1]  # Número de habitaciones

# Crear gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos reales
ax.scatter(dims, rooms, y.reshape(-1), c='r', marker='o', label='Datos Reales')
# Graficar puntos predichos
ax.scatter(dims, rooms, model_found.reshape(-1), c='b', marker='o', label='Modelo ajustado')

# Graficar la superficie ajustada por el modelo de regresión
# Crear una malla de puntos para las dimensiones y habitaciones
x_range = np.linspace(dims.min(), dims.max(), 20)
y_range = np.linspace(rooms.min(), rooms.max(), 20)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = thetas_optimal[0] + thetas_optimal[1] * x_mesh + thetas_optimal[2] * y_mesh

# Superficie de la predicción
ax.plot_surface(x_mesh, y_mesh, z_mesh, color='b', alpha=0.5)

# Etiquetas
ax.set_xlabel('Dimensión (m²)')
ax.set_ylabel('No. Habitaciones')
ax.set_zlabel('Precio (mil dólares)')

plt.legend()
plt.title('Regresión Lineal para Predecir Precio de Casas')

# Mostrar gráfico
plt.show()

