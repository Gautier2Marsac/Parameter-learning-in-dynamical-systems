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

# 3. Extraction et affichage des constantes estimées
coeffs = model.coefficients()

# Pour le système de Lorenz: dx/dt = σ(y - x), dy/dt = x(ρ - z) - y, dz/dt = xy - βz
# Extraction des coefficients (les indices dépendent de la structure de la bibliothèque)
# coeffs[i, j] où i est l'équation et j le terme de la bibliothèque

# σ estimé: coefficient de y dans l'équation x' (ou de x0 dans x1)
sigma_estimated = coeffs[0, 2]  # Coefficient de x1 dans (x0)'

# ρ estimé: coefficient de x dans l'équation y' 
rho_estimated = coeffs[1, 1]  # Coefficient de x0 dans (x1)'

# β estimé: coefficient de -z dans l'équation z'
beta_estimated = -coeffs[2, 3]  # Coefficient de x2 dans (x2)' (avec signe inversé)

print("\n" + "="*50)
print("CONSTANTES ESTIMÉES vs VRAIES VALEURS")
print("="*50)
print(f"σ (sigma) : estimé = {sigma_estimated:.3f}, vrai = 10.000")
print(f"ρ (rho)   : estimé = {rho_estimated:.3f}, vrai = 28.000")
print(f"β (beta)  : estimé = {beta_estimated:.3f}, vrai = {8/3:.3f}")
print("="*50)