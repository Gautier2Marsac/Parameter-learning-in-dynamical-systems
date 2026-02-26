import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Le système de Lorenz
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]

# 2. Paramètres de la Heatmap
PSEUDO_PERIOD = 0.9 

dts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
nb_loops = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20, 50, 100]

# Nouveaux paramètres pour la robustesse
n_trials = 10        # 10 essais par case suffisent pour lisser l'effet du bruit
noise_level = 0.5   # Ajout d'un bruit pour justifier les répétitions&&
epsilon = 1e-6  # Ta régularisation Julia
threshold = 0.1 # Seuil pour la parcimonie (SINDy)

error_matrix = np.zeros((len(dts), len(nb_loops)))

print(f"Calcul de la Heatmap robuste ({n_trials} essais/case, Bruit={noise_level})...")

for i, dt in enumerate(dts):
    print(f"Traitement du pas de temps dt = {dt}")
    for j, loops in enumerate(nb_loops):
        
        T_max = loops * PSEUDO_PERIOD
        t_eval = np.arange(0, T_max, dt)
        
        # Sécurité minimale de points
        if len(t_eval) < 15:
            error_matrix[i, j] = 100.0
            continue
            
        # 1. Génération de la trajectoire pure (une fois par configuration)
        sol = solve_ivp(lorenz, (0, T_max), [-8, 8, 27], t_eval=t_eval, rtol=1e-10, atol=1e-10)
        x_pure = sol.y.T
        
        # Liste pour stocker les erreurs de chaque essai pour cette configuration
        cell_errors = []
        
        # 2. Boucle de robustesse statistique
        for _ in range(n_trials):
            # Ajout d'un NOUVEAU bruit à chaque itération
            x_noisy = x_pure + noise_level * np.random.normal(size=x_pure.shape)
            
            # Calcul de la dérivée temporelle (approximée par les différences finies)
            x_dot = np.gradient(x_noisy, dt, axis=0)

            x_f, y_f, z_f = x_noisy[:, 0], x_noisy[:, 1], x_noisy[:, 2]
            # Ordre: [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]
            Theta = np.column_stack([
                np.ones_like(x_f), x_f, y_f, z_f, 
                x_f**2, x_f*y_f, x_f*z_f, y_f**2, y_f*z_f, z_f**2
            ])

            # M = F'F + eps*I
            M = Theta.T @ Theta + epsilon * np.eye(Theta.shape[1])
            # a_hat = M \ (F' * dXdt)
            Xi = np.linalg.solve(M, Theta.T @ x_dot)

            Xi[np.abs(Xi) < threshold] = 0


            #optimizer=ps.STLSQ(threshold=0.1)
            try:
                # Lorenz: dx/dt = 10(y-x), dy/dt = 28x - y - xz, dz/dt = xy - 8/3z
                s_est = Xi[2, 0]   # Coeff de y dans dx/dt
                r_est = Xi[1, 1]   # Coeff de x dans dy/dt
                b_est = -Xi[3, 2]  # Coeff de z dans dz/dt (index 3 car c'est le terme linéaire z)
                
                err = (abs(s_est-10)/10 + abs(r_est-28)/28 + abs(b_est-(8/3))/(8/3)) / 3 * 100
                cell_errors.append(min(err, 100.0))
            except:
                cell_errors.append(100.0)
        
        error_matrix[i, j] = np.mean(cell_errors)
        
        # 3. Moyenne des erreurs pour cette case précise
        error_matrix[i, j] = np.mean(cell_errors)

# --- Affichage de la Heatmap avec Seaborn ---
plt.figure(figsize=(12, 8))

ax = sns.heatmap(error_matrix, annot=True, fmt=".0f", cmap="YlOrRd", 
                 xticklabels=nb_loops, yticklabels=dts, 
                 cbar_kws={'label': 'Erreur Relative Moyenne (%)'})

plt.title(f"Robustesse de l'Identifiabilité de SINDy (Bruit={noise_level}, {n_trials} essais/case)")
plt.xlabel("Diversité Physique : Nombre de Boucles de Lorenz")
plt.ylabel("Résolution Temporelle : Pas de temps (dt)")

plt.yticks(rotation=0)
plt.tight_layout()
plt.show()