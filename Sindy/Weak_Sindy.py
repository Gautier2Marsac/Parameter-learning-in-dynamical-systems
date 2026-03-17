import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Le système de Lorenz
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]

# 2. Paramètres
PSEUDO_PERIOD = 0.9 
dts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
nb_loops = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20, 50, 100]
n_trials = 10
noise_level = 0.5
epsilon = 1e-6 
threshold = 0.1 

window_time = 0.6  # Durée temporelle de la fenêtre (en secondes). Lorenz a une pseudo-période de ~0.9s.
p_power = 4        # Exposant de la fonction test (augmente la douceur aux bords pour écraser le bruit)
n_windows = 200    # Nombre de fenêtres à extraire (plus on en a, meilleure est la moyenne statistique)

error_matrix = np.zeros((len(dts), len(nb_loops)))

print(f"Calcul Heatmap WSINDy ({n_trials} essais/case, Bruit={noise_level})...")

for i, dt in enumerate(dts):
    print(f"Traitement dt = {dt}")
    for j, loops in enumerate(nb_loops):
        T_max = loops * PSEUDO_PERIOD
        t_eval = np.arange(0, T_max, dt)
        
        if len(t_eval) < 20: # Il faut un minimum de points pour intégrer
            error_matrix[i, j] = 100.0
            continue
            
        sol = solve_ivp(lorenz, (0, T_max), [-8, 8, 27], t_eval=t_eval, rtol=1e-10, atol=1e-10)
        x_pure = sol.y.T
        cell_errors = []

        # --- PARAMÈTRES WSINDy ---
        # On définit une fenêtre d'intégration (ex: 0.4 secondes)
        L = max(8, int(window_time / dt)) 
        # Sécurité : si la trajectoire est trop courte pour la fenêtre, on prend 80% de la trajectoire
        if L > len(t_eval) - 2: 
            L = int(len(t_eval) * 0.8)
            
        if L < 5: # Si on a vraiment trop peu de points, on abandonne cette case
            error_matrix[i, j] = 100.0
            continue
        
        # Création de la fonction test v(t) et sa dérivée exacte v_dot(t)
        tau = np.arange(L) / (L - 1) # Temps normalisé de 0 à 1
        v = np.sin(np.pi * tau)**p_power
        v_dot = p_power * np.sin(np.pi * tau)**(p_power-1) * np.cos(np.pi * tau) * (np.pi / ((L - 1) * dt))

        for _ in range(n_trials):
            x_noisy = x_pure + noise_level * np.random.normal(size=x_pure.shape)
            
            # Bibliothèque Theta classique
            x_f, y_f, z_f = x_noisy[:, 0], x_noisy[:, 1], x_noisy[:, 2]
            Theta = np.column_stack([
                np.ones_like(x_f), x_f, y_f, z_f, 
                x_f**2, x_f*y_f, x_f*z_f, y_f**2, y_f*z_f, z_f**2
            ])

            # --- CONSTRUCTION DES MATRICES INTÉGRALES ---
            # On fait glisser notre fenêtre sur les données
            stride = max(1, (len(t_eval) - L) // n_windows) # On prend ~50 fenêtres pour aller vite
            windows_starts = np.arange(0, len(t_eval) - L, stride)
            
            V_integral = np.zeros((len(windows_starts), Theta.shape[1]))
            G_integral = np.zeros((len(windows_starts), 3))
            
            for idx, start in enumerate(windows_starts):
                x_win = x_noisy[start:start+L, :]
                theta_win = Theta[start:start+L, :]
                
                # G = - intégrale( X * v_dot dt )
                G_integral[idx, :] = -np.sum(x_win * v_dot[:, None], axis=0) * dt
                # V = intégrale( Theta * v dt )
                V_integral[idx, :] = np.sum(theta_win * v[:, None], axis=0) * dt

            # --- RÉSOLUTION STLSQ SUR LES INTÉGRALES ---
            M = V_integral.T @ V_integral + epsilon * np.eye(Theta.shape[1])
            Xi = np.linalg.solve(M, V_integral.T @ G_integral)

            # Boucle de seuillage (vrai STLSQ)
            for _ in range(3):
                small_inds = np.abs(Xi) < threshold
                Xi[small_inds] = 0
                for dim in range(3):
                    big_inds = ~small_inds[:, dim]
                    if np.sum(big_inds) > 0:
                        M_sub = V_integral[:, big_inds].T @ V_integral[:, big_inds] + epsilon * np.eye(np.sum(big_inds))
                        Xi[big_inds, dim] = np.linalg.solve(M_sub, V_integral[:, big_inds].T @ G_integral[:, dim])

            # --- EXTRACTION DES ERREURS ---
            try:
                s_est, r_est, b_est = Xi[2, 0], Xi[1, 1], -Xi[3, 2]
                err = (abs(s_est-10)/10 + abs(r_est-28)/28 + abs(b_est-(8/3))/(8/3)) / 3 * 100
                cell_errors.append(min(err, 100.0))
            except:
                cell_errors.append(100.0)
        
        error_matrix[i, j] = np.mean(cell_errors)

# --- Affichage ---
plt.figure(figsize=(10, 7))
sns.heatmap(error_matrix, annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=nb_loops, yticklabels=dts)
plt.title(f"WSINDy 'Handmade' (Bruit={noise_level})")
plt.xlabel("Boucles")
plt.ylabel("dt")
plt.show()