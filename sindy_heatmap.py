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
            
            # Application de SINDy


            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=2), 
                optimizer=ps.SR3(
                    regularizer='l1',       # Norme L1 douce (au lieu de 'l0' qui coupe brutalement)
                    reg_weight_lam=1e-3,     # Remplace le 'threshold' (force de la sélection)
                    relax_coeff_nu=0.5,     # Tolérance au bruit lors de l'optimisation
                    max_iter=2000           # Sécurité pour éviter les boucles infinies
                )
            )
            
            #optimizer=ps.STLSQ(threshold=0.1)
            try:
                model.fit(x_noisy, t=dt)
                coeffs = model.coefficients()
                
                if np.all(coeffs == 0):
                    cell_errors.append(100.0)
                else:
                    s, r, b = coeffs[0, 2], coeffs[1, 1], -coeffs[2, 3]
                    err = (abs(s-10)/10 + abs(r-28)/28 + abs(b-(8/3))/(8/3)) / 3 * 100
                    cell_errors.append(min(err, 100.0))
            except:
                cell_errors.append(100.0)
        
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