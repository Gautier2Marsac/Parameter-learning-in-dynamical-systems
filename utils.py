import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp


def plot_observations(observations, dt=None, var_names=None):
    var_names = ['$X_0$', '$X_1$', '$X_2$']

    time = np.arange(observations.shape[0])
    plt.figure(figsize=(15, 4))

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(time, observations[:, i])
        plt.title(f'Variable {var_names[i]}')
        plt.xlabel('Time step')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(observations, var_names=None):
    var_names = ['$X_0$', '$X_1$', '$X_2$']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory as a line
    ax.plot(observations[:, 0], observations[:, 1], observations[:, 2],
            color='blue', linewidth=1)

    # Optionally mark the start point
    ax.scatter(*observations[0], color='green', s=50, label='Start')

    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel(var_names[2])
    ax.set_title('3D Phase Space Trajectory')
    ax.legend()

    plt.tight_layout()
    plt.show()

def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]


def solve_lorenz(t_0 = 0, t_f=10, dt=0.001, X_0 = [-8, 8, 27]): 
    t_train = np.arange(t_0, t_f, dt)
    sol = solve_ivp(lorenz, (t_train[0], t_train[-1]), X_0, t_eval=t_train)
    observations = sol['y'].transpose(1,0)

    return observations

def evaluate(theta_estim, theta):
    erreur_rel = np.abs((theta_estim - theta) / theta) * 100
    print(f"L'erreur relative sur σ, ρ, β est de : {erreur_rel[0]}, {erreur_rel[1]}, {erreur_rel[2]}")
