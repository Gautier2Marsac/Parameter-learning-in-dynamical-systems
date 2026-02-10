import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt

def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]

# --- Paramètres du test ---
dts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.01, 0.005, 0.001]
n_trials = 100  # Nombre de répétitions par dt
noise_magnitude = 1.5 
true_vals = {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}

# Listes pour stocker les moyennes finales
avg_err_sigma, avg_err_rho, avg_err_beta, avg_err_mre = [], [], [], []

print(f"Lancement de l'étude statistique ({n_trials} essais par dt)...")

for dt in dts:
    trial_sigma, trial_rho, trial_beta = [], [], []
    
    # 1. Génération de la trajectoire "propre" (une seule fois par dt pour gagner du temps)
    t_train = np.arange(0, 10, dt)
    x0_train = [-8, 8, 27]
    sol = solve_ivp(lorenz, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)
    x_pure = sol.y.T

    for _ in range(n_trials):
        # 2. Ajout d'un nouveau bruit aléatoire à chaque essai
        x_noisy = x_pure + noise_magnitude * np.random.normal(size=x_pure.shape)

        # 3. Application de SINDy
        model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), 
                         optimizer=ps.STLSQ(threshold=0.1))
        
        try:
            model.fit(x_noisy, t=dt)
            coeffs = model.coefficients()
            
            # Extraction et calcul d'erreur
            s_est = coeffs[0, 2]
            r_est = coeffs[1, 1]
            b_est = -coeffs[2, 3]
            
            trial_sigma.append(abs(s_est - 10) / 10 * 100)
            trial_rho.append(abs(r_est - 28) / 28 * 100)
            trial_beta.append(abs(b_est - (8/3)) / (8/3) * 100)
        except:
            continue # On ignore les échecs de convergence

    # Calcul des moyennes pour ce dt (si au moins un essai a réussi)
    if trial_sigma:
        m_s = np.mean(trial_sigma)
        m_r = np.mean(trial_rho)
        m_b = np.mean(trial_beta)
        avg_err_sigma.append(m_s)
        avg_err_rho.append(m_r)
        avg_err_beta.append(m_b)
        avg_err_mre.append((m_s + m_r + m_b) / 3)
    else:
        for l in [avg_err_sigma, avg_err_rho, avg_err_beta, avg_err_mre]:
            l.append(np.nan)

# --- Affichage des résultats moyens ---
plt.figure(figsize=(12, 7))
plt.semilogx(dts, avg_err_sigma, 'r--o', label='Moyenne $\sigma$ (Sigma)', alpha=0.6)
plt.semilogx(dts, avg_err_rho, 'g--o', label='Moyenne $\\rho$ (Rho)', alpha=0.6)
plt.semilogx(dts, avg_err_beta, 'b--o', label='Moyenne $\\beta$ (Beta)', alpha=0.6)
plt.semilogx(dts, avg_err_mre, 'ko-', label='MRE Globale Moyenne', linewidth=2.5)

plt.gca().invert_xaxis()
plt.xlabel('Pas de temps (dt)')
plt.ylabel('Erreur Relative Moyenne (%)')
plt.title(f'Robustesse de SINDy sur Lorenz-63 (Moyenne sur {n_trials} essais, Bruit={noise_magnitude})')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()