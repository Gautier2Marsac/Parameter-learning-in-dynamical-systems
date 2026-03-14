import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from tqdm import tqdm
from utils import plot_3d_trajectory, plot_observations
from utils import solve_lorenz
from utils import evaluate
from utils import plot_estim_evolution
from utils import sgd_update
import optuna
import configparser
from utils import compute_snr



def f(X, theta):
    return np.array([theta[0]*X[1] - theta[0]*X[0],
                     theta[1]*X[0] - X[0]*X[2] - X[1],
                     X[0]*X[1] - theta[2]*X[2]])

def runge_kutta4(f, X_n, theta, h):
    k1 = f(X_n, theta)
    k2 = f(X_n + h*k1/2, theta)
    k3 = f(X_n + h*k2/2, theta)
    k4 = f(X_n + h*k3, theta)
    return X_n + h*(k1 + 2*k2 + 2*k3 + k4)/6



X_0 = np.array([-8.0, 8.0, 27.0], dtype=np.float64)
theta = np.array([10.0, 28.0, 8.0/3.0], dtype=np.float64)

t_0, t_f = 0, 2
h = 1e-4
N = int((t_f-t_0)/h)

observations = np.zeros((N, 3))
observations[0] = np.array(X_0)

for n in range(N-1):
    observations[n+1] = runge_kutta4(f, observations[n], theta, h)

observations_solver = solve_lorenz(t_0=t_0, t_f=t_f, dt=h, X_0=X_0)


def compute_grad(f, methode, theta_estim, A, estim_obs, n):
    sum_term = np.dot(estim_obs[:n], A).sum(axis=0)   

    X_nm1 = estim_obs[n-1]
    X_n   = estim_obs[n] 
    
    error = methode(f, X_nm1, theta_estim, h) -  X_n
    # error = h * f(X_nm1, theta_estim) + X_nm1 -  X_n

    grad = sum_term * error
    return grad

def decay_lr(lr, epoch, decay = 1e-4): 
    return lr


def decay_grad(grads, current_grad_idx, clip_value=np.array([25,10,10]), window=40, decay=0.8):

    grad = grads[current_grad_idx]

    # --- soft clipping ---
    grad = clip_value * np.tanh(grad / clip_value)
    grads[current_grad_idx] = grad

    # --- window selection ---
    start = max(0, current_grad_idx - window + 1)
    window_grads = grads[start:current_grad_idx + 1]

    # --- exponential decay weights ---
    n = len(window_grads)
    weights = decay ** np.arange(n)[::-1]
    weights = weights / np.sum(weights)

    # --- weighted average ---
    smoothed_grad = np.sum(window_grads * weights[:, None], axis=0)

    # grad_norm = smoothed_grad / (np.linalg.norm(smoothed_grad) + 1e-8)

    return smoothed_grad


def main(window = 40, 
         decay = 0.8, 
         epochs = 2,
         var = 0.01,
         lr = np.array([1e-3, 1e-3, 1e-3]), 
         clip_value = np.array([25,10,10])): 
    
    theta_estim = np.array([-5.0, -3.0, 10.0])

    nbr_epochs = int(epochs * N)
    A = np.array([[-1, 1, 0],
                [1 , 0, 0],
                [0 , 0, -1]])

    estim_obs = observations.copy() + np.random.normal(0, var, (N, 3))

    theta_estim_vec = np.zeros((nbr_epochs, 3))
    theta_estim_vec[0] = theta_estim

    grads = np.zeros((nbr_epochs, 3))

    for epoch in tqdm(range(1, nbr_epochs)):
        n = epoch % N
        grad = compute_grad(f, runge_kutta4, theta_estim, A, estim_obs, n)
        grads[epoch, :] = grad
        grad = decay_grad(grads, epoch, clip_value, window, decay)
        theta_estim = sgd_update(theta_estim, grad, lr)
        lr = decay_lr(lr, epoch)
        theta_estim_vec[epoch] = theta_estim
    
    return theta_estim, theta_estim_vec, grads


config = configparser.ConfigParser()
config.read('config.ini')
noise_var = config['Optimizer'].getfloat('var')

snr = compute_snr(observations, noise_var)
print(f"SNR (dB) for each dimension: {snr}")
print(f"Average SNR (dB): {np.mean(snr):.2f}")



def objective(trial):

    window = trial.suggest_int('window', 10, 100)  
    decay = trial.suggest_float('decay', 0.1, 0.99) 
    nbr_epochs = trial.suggest_float('nbr_epochs', 0.5, 3) 

    lr = [
        trial.suggest_float('lr_0', 1e-4, 1e-3, log=True),
        trial.suggest_float('lr_1', 1e-4, 1e-3, log=True),
        trial.suggest_float('lr_2', 1e-4, 1e-2, log=True)
    ]
    clip_value = [
        trial.suggest_int('clip_0', 1, 40),
        trial.suggest_int('clip_1', 1, 20),
        trial.suggest_int('clip_2', 1, 20)
    ]

    theta_estim, theta_estim_vec, grads = main(window, decay, nbr_epochs, noise_var, np.array(lr), np.array(clip_value))
    evaluate(theta_estim, theta)

    # result = np.mean((theta_estim - theta)**2)
    result = np.mean(((theta_estim - theta)/theta)**2)*100

    # result = np.var(grads)
    # result = np.mean(np.abs(grads))

    # # this dosn't work worked the better model does not have the min of this
    # result = np.mean(np.abs(np.diff(theta_estim_vec, axis=0))) 

    # result = np.mean(np.abs(grads[-50:]))
    #result = np.mean(np.linalg.norm(grads, axis=1))
    # print(np.mean(((theta_estim - theta)/theta)**2)*100)

    return result


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100)


print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")