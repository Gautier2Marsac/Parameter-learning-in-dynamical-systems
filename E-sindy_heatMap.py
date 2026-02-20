import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore") # Pour garder la console propre

# 1. Le système de Lorenz
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]

# 2. Paramètres de la Heatmap
PSEUDO_PERIOD = 0.9 

# J'ai légèrement réduit la grille pour que le test ne prenne pas 2 heures, 
# E-SINDy étant plus gourmand en calcul (il lance X modèles par case).
# Vous pourrez remettre la grille complète pour le rapport final.
dts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
nb_loops = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20, 50, 100]

noise_level = 0.5   
n_trials = 2  # On réduit à 2 essais globaux car E-SINDy fait déjà le travail statistique

error_matrix = np.zeros((len(dts), len(nb_loops)))

print(f"Calcul de la Heatmap avec E-SINDy (Bruit={noise_level})...")

for i, dt in enumerate(dts):
    print(f"Traitement du pas de temps dt = {dt}")
    for j, loops in enumerate(nb_loops):
        
        T_max = loops * PSEUDO_PERIOD
        t_eval = np.arange(0, T_max, dt)
        
        if len(t_eval) < 15:
            error_matrix[i, j] = 100.0
            continue
            
        sol = solve_ivp(lorenz, (0, T_max), [-8, 8, 27], t_eval=t_eval, rtol=1e-10, atol=1e-10)
        x_pure = sol.y.T
        
        cell_errors = []
        
        for _ in range(n_trials):
            x_noisy = x_pure + noise_level * np.random.normal(size=x_pure.shape)
            
            # --- IMPLÉMENTATION DE E-SINDY ---
            # 1. On définit l'optimiseur de base (votre champion STLSQ)
            base_optimizer = ps.STLSQ(threshold=0.1)
            
            # 2. On l'enveloppe dans l'EnsembleOptimizer
            # n_models=20 : On crée un "jury" de 20 STLSQ.
            # bagging=True : Chaque membre du jury voit un sous-échantillon aléatoire des données.
            ensemble_optimizer = ps.EnsembleOptimizer(
                opt=base_optimizer, 
                bagging=True, 
                n_models=20 
            )
            
            # 3. On crée le modèle
            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=2), 
                optimizer=ensemble_optimizer
            )
            
            try:
                model.fit(x_noisy, t=dt)
                
                # PySINDy calcule automatiquement la médiane des 20 modèles !
                coeffs = model.coefficients() 
                
                if np.all(coeffs == 0):
                    cell_errors.append(100.0)
                else:
                    s, r, b = coeffs[0, 2], coeffs[1, 1], -coeffs[2, 3]
                    err = (abs(s-10)/10 + abs(r-28)/28 + abs(b-(8/3))/(8/3)) / 3 * 100
                    cell_errors.append(min(err, 100.0))
            except:
                cell_errors.append(100.0)
        
        error_matrix[i, j] = np.mean(cell_errors)

# --- Affichage de la Heatmap ---
plt.figure(figsize=(10, 7))

ax = sns.heatmap(error_matrix, annot=True, fmt=".0f", cmap="YlOrRd", 
                 xticklabels=nb_loops, yticklabels=dts, 
                 cbar_kws={'label': 'Erreur Relative Moyenne (%)'})

plt.title(f"Identifiabilité avec E-SINDy (STLSQ Bagging, Bruit={noise_level})")
plt.xlabel("Diversité Physique : Nombre de Boucles de Lorenz")
plt.ylabel("Résolution Temporelle : Pas de temps (dt)")

plt.yticks(rotation=0)
plt.tight_layout()
plt.show()