import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt

# 1. Définition du système (Lorenz-63)
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]]

# 2. Paramètres de l'expérience rigoureuse
# On fixe l'Horizon Physique (T_max) pour que tous les modèles voient la même "quantité" de physique
T_MAX = 200 
dts = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.005,0.001,0.00001] # Gamme de pas de temps
n_trials = 50       # Nombre d'essais pour la robustesse statistique
noise_level =  0  # Bruit fixe pour évaluer la résistance du dt

stats = {dt: [] for dt in dts}

print(f"Lancement du protocole à Horizon Fixe (T={T_MAX}s)...")

for dt in dts:
    print(dt)
    # Pour chaque dt, on génère la grille temporelle couvrant exactement T_MAX
    t_eval = np.arange(0, T_MAX, dt)
    
    # Vérification de sécurité : si dt est trop grand, on risque d'avoir trop peu de points
    if len(t_eval) < 20: 
        print(f"Warning: dt={dt} donne seulement {len(t_eval)} points. Ignoré.")
        stats[dt] = [np.nan] * n_trials
        continue

    # Génération de la vérité terrain (Trajectoire pure sur T_MAX)
    # On intègre une seule fois par dt pour avoir la grille exacte
    sol = solve_ivp(lorenz, (0, T_MAX), [-8, 8, 27], t_eval=t_eval, rtol=1e-10, atol=1e-10)
    x_pure = sol.y.T

    for _ in range(n_trials):
        # Ajout du bruit (Indépendant du dt, dépendant de la mesure)
        x_noisy = x_pure + noise_level * np.random.normal(size=x_pure.shape)

        # SINDy configuration
        model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), 
                         optimizer=ps.STLSQ(threshold=0.01))
        
        try:
            model.fit(x_noisy, t=dt)
            coeffs = model.coefficients()
            
            # Calcul de l'erreur relative moyenne sur les 3 paramètres
            s, r, b = coeffs[0, 2], coeffs[1, 1], -coeffs[2, 3]
            err_s = abs(s - 10)/10
            err_r = abs(r - 28)/28
            err_b = abs(b - 8/3)/(8/3)
            
            # Score global (MRE)
            stats[dt].append((err_s + err_r + err_b) / 3 * 100) # En pourcentage
        except:
            stats[dt].append(100.0) # Échec considéré comme 100% d'erreur

# --- Visualisation ---
means = [np.nanmean(stats[dt]) for dt in dts]
stds = [np.nanstd(stats[dt]) for dt in dts]

plt.figure(figsize=(10, 6))
plt.errorbar(dts, means, yerr=stds, fmt='-o', color='teal', ecolor='gray', capsize=5, label='Erreur MRE (%)')
plt.xscale('log')
plt.gca().invert_xaxis() # Du plus grand dt (gauche) au plus précis (droite)
plt.xlabel('Pas de temps (dt) [Log scale]')
plt.ylabel('Erreur Moyenne (%)')
plt.title(f'Performance de SINDy à Horizon Physique Constant (T={T_MAX}s)')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()