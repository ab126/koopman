from src.core import gen_max_theta_data, identify_sys_multiple_trajectories, linearize_upright_dynamics, lqr_F_caller, lqr_F_caller, simulate
from src.plotting import animate, animate_point_mass, plot_results, save_animation
import numpy as np
from IPython.display import display, HTML

# ----------------------------
# Parameters
# ----------------------------
M = 2.0      # slab mass
m = 1.0      # rod mass
l = 1.0      # rod length
g = 9.81

# ----------------------------
# Input force function F(t)
# ----------------------------
def fb_F(t, y):
    # Example: simple stabilizing feedback 
    k_theta = 20
    k_theta_dot = 5
    return -k_theta * y[2] - k_theta_dot * y[3]

def zero_F(t, y):
    # No force
    return 0

x0 = np.array([0, 0, 0.8, 0])
t, x, x_dot, theta, theta_dot, u = simulate(zero_F, M, m, g, l, y0=x0) # Normalized to m=g=l=1


# ----------------------------
# Generate Small Angle Trajectories
# ----------------------------
num_points = 1000
t_span = (0, 2)
n_repeats = 100
sigma = 1
theta_max = np.pi / 20
t_all, X_all, F_all = gen_max_theta_data(M, m, g, l, sigma=sigma, theta_max=theta_max, t_span=t_span, num_points=num_points, n_repeats=n_repeats)

# ----------------------------
# Identify Linear System
# ----------------------------
A_s, B_s = identify_sys_multiple_trajectories(t_all, X_all, F_all, m, g, l, model_type="continuous")#, lift=lambda y: y)

# ----------------------------
# Linear Quadratic Regulator
# ----------------------------
Q = np.diag([1, 1, 50, 10])
R = np.array([[0.1]])
x0 = np.array([0, 0, 0.1, 0])

opt_F = lqr_F_caller(
    A_s, B_s,
    m=m, g=g, l=l,
    Q=Q,
    R=R,
    umax=10,
    y_ref=np.zeros(4),
)

t, x, x_dot, theta, theta_dot, u = simulate(opt_F, M, m, g, l, y0=x0)

# ----------------------------
# Simulate and Plot Results
# ----------------------------
ani = animate(t, x, theta, l)
#ani = animate_point_mass(t, x, theta, l, u)
#HTML(ani.to_jshtml())

save_animation(ani, 'saves/lqr_small_angle.gif') 

