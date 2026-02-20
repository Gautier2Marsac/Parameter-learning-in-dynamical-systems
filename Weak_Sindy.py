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

dts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
nb_loops = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20, 50, 100]

noise_level = 0.5   
n_trials = 2  # 2 essais suffisent pour voir la puissance de la méthode faible

error_matrix = np.zeros((len(dts), len(nb_loops)))

print(f"Calcul de la Heatmap avec Weak SINDy (Bruit={noise_level})...")

for i, dt in enumerate(dts):
    print(f"Traitement du pas de temps dt = {dt}")
    for j, loops in enumerate(nb_loops):
        
        T_max = loops * PSEUDO_PERIOD
        t_eval = np.arange(0, T_max, dt)
        
        # Sécurité pour Weak SINDy : nécessite un minimum de points pour intégrer
        if len(t_eval) < 20:
            error_matrix[i, j] = 100.0
            continue
            
        # Génération de la vérité terrain
        sol = solve_ivp(lorenz, (0, T_max), [-8, 8, 27], t_eval=t_eval, rtol=1e-10, atol=1e-10)
        x_pure = sol.y.T
        
        cell_errors = []
        
        for _ in range(n_trials):
            # Ajout du bruit
            x_noisy = x_pure + noise_level * np.random.normal(size=x_pure.shape)
            
# --- L'ULTIME COMBO : ENSEMBLE + WEAK SINDY ---
            t_grid = t_eval
            
            # 1. La barrière anti-bruit (Intégration Faible)
            weak_lib = ps.WeakPDELibrary(
                function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
                spatiotemporal_grid=t_grid,
                is_uniform=True,
                K=100 # 100 domaines d'intégration
            )
            
            # 2. Le champion de base
            base_optimizer = ps.STLSQ(threshold=0.1)
            
            # 3. La barrière anti-hallucinations (Bagging)
            ensemble_optimizer = ps.EnsembleOptimizer(
                opt=base_optimizer, 
                bagging=True, 
                n_models=20 
            )
            
            # 4. Le modèle fusionné
            model = ps.SINDy(
                feature_library=weak_lib, 
                optimizer=ensemble_optimizer 
            )
            
            try:
                # Entraînement
                model.fit(x_noisy, t=dt) 
                
                # Récupération de la médiane des 20 modèles faibles
                coeffs = model.coefficients() 
                
                if np.all(coeffs == 0):
                    cell_errors.append(100.0)
                else:
                    s = coeffs[0, 1]     
                    r = coeffs[1, 0]     
                    b = -coeffs[2, 2]    
                    
                    err = (abs(s-10)/10 + abs(r-28)/28 + abs(b-(8/3))/(8/3)) / 3 * 100
                    cell_errors.append(min(err, 100.0))
                        
            except Exception as e:
                cell_errors.append(100.0)
        
        # Moyenne des erreurs pour la case
        error_matrix[i, j] = np.mean(cell_errors)

# --- Affichage de la Heatmap avec Seaborn ---
plt.figure(figsize=(12, 8))

ax = sns.heatmap(error_matrix, annot=True, fmt=".0f", cmap="YlOrRd", 
                 xticklabels=nb_loops, yticklabels=dts, 
                 cbar_kws={'label': 'Erreur Relative Moyenne (%)'})

plt.title(f"Identifiabilité avec Weak SINDy (Formulation Faible, Bruit={noise_level})")
plt.xlabel("Diversité Physique : Nombre de Boucles de Lorenz")
plt.ylabel("Résolution Temporelle : Pas de temps (dt)")

plt.yticks(rotation=0)
plt.tight_layout()
plt.show()