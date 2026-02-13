import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. Génération des données expérimentales
# ==========================================
def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Paramètres de simulation
dt = 0.01
t_end = 3.0
t_eval = np.arange(0, t_end, dt)
initial_state = [-8.0, 7.0, 27.0]

# Résolution numérique (Vérité terrain)
solution = solve_ivp(lorenz_system, [0, t_end], initial_state, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
t_data = solution.t.reshape(-1, 1)
X_data = solution.y.T  # (N, 3)

# --- NORMALISATION DES DONNÉES ---
# Normaliser le temps à [0, 1]
t_min, t_max = t_data.min(), t_data.max()
t_norm = (t_data - t_min) / (t_max - t_min)

# Normaliser les états (moyenne=0, std=1)
X_mean = X_data.mean(axis=0, keepdims=True)
X_std = X_data.std(axis=0, keepdims=True)
X_norm = (X_data - X_mean) / X_std

# Facteur d'échelle temporel pour corriger les dérivées : d(X_norm)/d(t_norm) = (t_max-t_min)/X_std * dX/dt
time_scale = (t_max - t_min)  # dt_norm = dt / time_scale

# Conversion en tenseurs PyTorch
t_tensor = torch.tensor(t_norm, dtype=torch.float32)
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
X_mean_t = torch.tensor(X_mean, dtype=torch.float32)
X_std_t = torch.tensor(X_std, dtype=torch.float32)

# ==========================================
# 2. Définition de l'architecture du PINN
# ==========================================
class LorenzPINN(nn.Module):
    def __init__(self):
        super(LorenzPINN, self).__init__()
        
        # Réseau plus large : t_norm -> [X_norm, Y_norm, Z_norm]
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
        
        # Initialisation Xavier pour une meilleure convergence
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Paramètres physiques à découvrir (initialisés plus proches de l'ordre de grandeur)
        self.sigma = nn.Parameter(torch.tensor(5.0))
        self.rho   = nn.Parameter(torch.tensor(15.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

    def forward(self, t):
        return self.net(t)

    def compute_loss(self, t, X_true_norm, X_mean, X_std, time_scale):
        t = t.clone().detach().requires_grad_(True)
        
        # Prédictions normalisées
        X_pred_norm = self.forward(t)
        
        # 1. DATA LOSS
        loss_data = torch.mean((X_pred_norm - X_true_norm)**2)
        
        # Dé-normaliser pour la physique : X_real = X_pred_norm * X_std + X_mean
        x_real = X_pred_norm[:, 0:1] * X_std[0, 0] + X_mean[0, 0]
        y_real = X_pred_norm[:, 1:2] * X_std[0, 1] + X_mean[0, 1]
        z_real = X_pred_norm[:, 2:3] * X_std[0, 2] + X_mean[0, 2]
        
        # Dérivées par rapport au temps normalisé, puis correction : dX/dt_real = dX/dt_norm / time_scale
        dx_dt_norm = torch.autograd.grad(x_real, t, grad_outputs=torch.ones_like(x_real), create_graph=True)[0]
        dy_dt_norm = torch.autograd.grad(y_real, t, grad_outputs=torch.ones_like(y_real), create_graph=True)[0]
        dz_dt_norm = torch.autograd.grad(z_real, t, grad_outputs=torch.ones_like(z_real), create_graph=True)[0]
        
        # dX/dt_real = dX/dt_norm / time_scale  (car t_real = t_norm * time_scale + t_min)
        dx_dt = dx_dt_norm / time_scale
        dy_dt = dy_dt_norm / time_scale
        dz_dt = dz_dt_norm / time_scale
        
        # 2. PHYSICS LOSS (équations de Lorenz)
        f_x = dx_dt - self.sigma * (y_real - x_real)
        f_y = dy_dt - (x_real * (self.rho - z_real) - y_real)
        f_z = dz_dt - (x_real * y_real - self.beta * z_real)
        
        loss_physics = torch.mean(f_x**2) + torch.mean(f_y**2) + torch.mean(f_z**2)
        
        # 3. CONDITION INITIALE (renforcer le point de départ)
        X_pred_0 = self.forward(t[0:1])
        x0_real = X_pred_0[:, 0:1] * X_std[0, 0] + X_mean[0, 0]
        y0_real = X_pred_0[:, 1:2] * X_std[0, 1] + X_mean[0, 1]
        z0_real = X_pred_0[:, 2:3] * X_std[0, 2] + X_mean[0, 2]
        loss_ic = (x0_real - (-8.0))**2 + (y0_real - 7.0)**2 + (z0_real - 27.0)**2
        loss_ic = loss_ic.mean()
        
        return loss_data, loss_physics, loss_ic

# ==========================================
# 3. Boucle d'entraînement
# ==========================================
model = LorenzPINN()

# Optimiseurs séparés : LR plus élevé pour les paramètres physiques
optimizer_nn = torch.optim.Adam(model.net.parameters(), lr=1e-3)
optimizer_phys = torch.optim.Adam([model.sigma, model.rho, model.beta], lr=5e-3)

# Schedulers pour réduire le LR progressivement
scheduler_nn = torch.optim.lr_scheduler.StepLR(optimizer_nn, step_size=3000, gamma=0.5)
scheduler_phys = torch.optim.lr_scheduler.StepLR(optimizer_phys, step_size=3000, gamma=0.5)

# --- PHASE 1 : Pré-entraînement du réseau sur les données (sans physique) ---
epochs_pretrain = 2000
print("Phase 1 : Pré-entraînement sur les données...")
optimizer_pretrain = torch.optim.Adam(model.net.parameters(), lr=1e-3)
for epoch in range(epochs_pretrain):
    optimizer_pretrain.zero_grad()
    X_pred = model(t_tensor)
    loss = torch.mean((X_pred - X_tensor)**2)
    loss.backward()
    optimizer_pretrain.step()
    if epoch % 500 == 0:
        print(f"  Pretrain Epoch {epoch} | Data Loss: {loss.item():.6f}")

# --- PHASE 2 : Entraînement complet (données + physique) ---
epochs = 15000
# Poids de la physics loss (augmente progressivement)
lambda_phys_start = 1e-4
lambda_phys_end = 1e-1
lambda_ic = 1.0

print("\nPhase 2 : Entraînement avec physique...")
for epoch in range(epochs):
    optimizer_nn.zero_grad()
    optimizer_phys.zero_grad()
    
    # Augmentation progressive du poids physique (curriculum)
    progress = min(epoch / (epochs * 0.5), 1.0)
    lambda_phys = lambda_phys_start + (lambda_phys_end - lambda_phys_start) * progress
    
    loss_data, loss_physics, loss_ic = model.compute_loss(
        t_tensor, X_tensor, X_mean_t, X_std_t, time_scale
    )
    
    loss = loss_data + lambda_phys * loss_physics + lambda_ic * loss_ic
    
    loss.backward()
    optimizer_nn.step()
    optimizer_phys.step()
    scheduler_nn.step()
    scheduler_phys.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f} | Data: {loss_data.item():.4f} | "
              f"Phys: {loss_physics.item():.1f} | λ_phys: {lambda_phys:.5f} | "
              f"σ={model.sigma.item():.2f}  ρ={model.rho.item():.2f}  β={model.beta.item():.2f}")

# --- PHASE 3 : Affinage avec L-BFGS ---
print("\nPhase 3 : Affinage avec L-BFGS...")
lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=20, history_size=50,
    line_search_fn='strong_wolfe'
)
lambda_phys_final = lambda_phys_end

for i in range(200):
    def closure():
        lbfgs_optimizer.zero_grad()
        ld, lp, lic = model.compute_loss(t_tensor, X_tensor, X_mean_t, X_std_t, time_scale)
        loss = ld + lambda_phys_final * lp + lambda_ic * lic
        loss.backward()
        return loss
    lbfgs_optimizer.step(closure)
    if i % 50 == 0:
        print(f"  L-BFGS iter {i} | σ={model.sigma.item():.2f}  ρ={model.rho.item():.2f}  β={model.beta.item():.2f}")

print("\n--- Résultats Finaux ---")
print(f"Vérité terrain : Sigma=10.00, Rho=28.00, Beta=2.67")
print(f"Estimations    : Sigma={model.sigma.item():.2f}, Rho={model.rho.item():.2f}, Beta={model.beta.item():.2f}")

# ==========================================
# 4. Visualisation
# ==========================================
with torch.no_grad():
    X_pred_norm = model(t_tensor).numpy()
    X_pred = X_pred_norm * X_std + X_mean

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ['X', 'Y', 'Z']
for i in range(3):
    axes[i].plot(t_data, X_data[:, i], 'b-', label='Vérité terrain', alpha=0.7)
    axes[i].plot(t_data, X_pred[:, i], 'r--', label='PINN', alpha=0.7)
    axes[i].set_ylabel(labels[i])
    axes[i].legend()
axes[2].set_xlabel('Temps')
fig.suptitle(f'σ={model.sigma.item():.2f}, ρ={model.rho.item():.2f}, β={model.beta.item():.2f}')
plt.tight_layout()
plt.show()