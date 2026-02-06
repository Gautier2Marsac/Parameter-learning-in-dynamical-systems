import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

# 1. Génération du Dataset (Vérité terrain)
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]

dt = 0.01
t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
sol = solve_ivp(lorenz, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)
x_train = sol.y.T

# 2. Application de SINDy
# On définit une bibliothèque polynomiale de degré 2 (car Lorenz a des termes type XZ et XY)
feature_library = ps.PolynomialLibrary(degree=2)
optimizer = ps.STLSQ(threshold=0.1) # Algorithme de seuillage itératif

model = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer
)

model.fit(x_train, t=dt)
model.print()