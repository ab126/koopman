import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

def plot_results(t, x, x_dot, theta, theta_dot, u):
    """Plots the state trajectories and control input over time."""

    plt.figure(figsize=(12, 8))

    plt.subplot(3,1,1)
    plt.plot(t, x, label="x")
    plt.plot(t, theta, label="theta")
    plt.legend()
    plt.title("States")

    plt.subplot(3,1,2)
    plt.plot(t, x_dot, label="x_dot")
    plt.plot(t, theta_dot, label="theta_dot")
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t, u, label="F(t)")
    plt.legend()
    plt.title("Input")

    plt.tight_layout()
    plt.show()
    
# ----------------------------
# Animation
# ----------------------------
def animate(t, x, theta, l):

    fig, ax = plt.subplots()

    # Dynamic limits (prevents object leaving frame)
    ax.set_xlim(np.min(x) - 1, np.max(x) + 1)
    ax.set_ylim(-.5, 1.5)
    ax.set_aspect('equal')
    ax.grid()

    # Slab (block)
    slab_width = 0.8
    slab_height = 0.3
    slab = Rectangle((-slab_width/2, 0), slab_width, slab_height, color='black')
    ax.add_patch(slab)

    # Rod (thick rectangle)
    rod_width = 0.1
    rod = Rectangle((- rod_width/2, slab_height), rod_width, l, color='red')
    ax.add_patch(rod)

    def init():
        return slab, rod

    def update(frame):
        xp = x[frame]
        th = theta[frame]

        # --- Update slab ---
        slab.set_xy((xp - slab_width/2, 0))

        # --- Rod geometry ---
        # Rectangle is defined at base, we rotate it
        rod.set_width(rod_width)
        rod.set_height(l)

        # Move base of rod to pivot point
        rod.set_xy((xp - rod_width/2, slab_height))

        # Apply rotation about base
        trans = (
            transforms.Affine2D()
            .rotate_around(xp, slab_height, -th)  # minus sign for correct direction
            + ax.transData
        )
        rod.set_transform(trans)

        return slab, rod

    interval = (t[-1] - t[0]) / len(t) * 1000  # Convert to milliseconds
    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval, # 30
        blit=True   # False
    )

    plt.title("Slab + Rigid Rod Animation")

    # prevent garbage collection
    global anim_ref
    anim_ref = ani
    plt.close(fig)
    return ani

def animate_point_mass(t, x, theta, l, F, case_name='', frame_step = 2, interval=None):
    """ Animates the point mass inverted pendulum system """

    frame_indices = np.arange(0, len(t), frame_step)

    cart_width = 0.4
    cart_height = 0.2
    wheel_radius = 0.05

    fig, ax = plt.subplots(figsize=(9, 5))

    # Keep vertical limits fixed, since the pendulum length does not change
    ax.set_ylim(-0.3, 1.4)
    ax.set_aspect('equal')
    ax.set_title(f"Inverted Pendulum {case_name}")
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Vertical position")
    ax.grid(True)

    # Ground line will also be updated each frame so it spans the current camera window
    ground_y = -0.1
    ground_line, = ax.plot([], [], linewidth=2)

    # Artists we will update frame-by-frame
    cart_body, = ax.plot([], [], linewidth=3)
    left_wheel, = ax.plot([], [], marker='o', markersize=8)
    right_wheel, = ax.plot([], [], marker='o', markersize=8)
    rod_line, = ax.plot([], [], linewidth=3)
    bob_point, = ax.plot([], [], marker='o', markersize=10)
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va='top')

    def init():
        ground_line.set_data([], [])
        cart_body.set_data([], [])
        left_wheel.set_data([], [])
        right_wheel.set_data([], [])
        rod_line.set_data([], [])
        bob_point.set_data([], [])
        info_text.set_text("")
        return ground_line, cart_body, left_wheel, right_wheel, rod_line, bob_point, info_text

    def update(frame_idx):
        i = frame_indices[frame_idx]

        xi = x[i]
        thetai = theta[i]   # raw angle for geometry

        # -----------------------------
        # Moving camera window
        # -----------------------------
        view_half_width = 2.0
        ax.set_xlim(xi - view_half_width, xi + view_half_width)

        # Update the ground line so it fills the current view
        ground_line.set_data(
            [xi - view_half_width, xi + view_half_width],
            [ground_y, ground_y]
        )

        # -----------------------------
        # Cart body
        # -----------------------------
        cart_left = xi - cart_width / 2
        cart_right = xi + cart_width / 2
        cart_bottom = 0.0
        cart_top = cart_height

        cart_xs = [cart_left, cart_right, cart_right, cart_left, cart_left]
        cart_ys = [cart_bottom, cart_bottom, cart_top, cart_top, cart_bottom]
        cart_body.set_data(cart_xs, cart_ys)

        # -----------------------------
        # Wheels
        # -----------------------------
        left_wheel.set_data([xi - cart_width / 4], [ground_y + wheel_radius])
        right_wheel.set_data([xi + cart_width / 4], [ground_y + wheel_radius])

        # -----------------------------
        # Pendulum geometry
        # -----------------------------
        pivot_x = xi
        pivot_y = cart_top

        # theta = 0 upright, clockwise positive
        bob_x = pivot_x + l * np.sin(thetai)
        bob_y = pivot_y + l * np.cos(thetai)

        rod_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
        bob_point.set_data([bob_x], [bob_y])

        # -----------------------------
        # On-figure text
        # -----------------------------
        info_text.set_text(
            f"t = {t[i]:.2f} s\n"
            f"x = {x[i]:.2f} m\n"
            f"theta = {theta[i]/np.pi * 180:.2f} deg\n"
            f"F = {F[i]:.2f} N"
        )

        return ground_line, cart_body, left_wheel, right_wheel, rod_line, bob_point, info_text
    
    if interval is None:
        interval = (t[-1] - t[0]) / len(t) * 1000  # Convert to milliseconds

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=interval * frame_step,
        blit=True
    )
    plt.close(fig)
    return anim

def save_animation(anim, filename):
    """Saves the animation to a file."""
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    anim.save(filename, writer='ffmpeg', fps=30)

# For concatenating generated trajectories
def concat_first_axis(arr_list):
    arr_2d = [
        a if a.ndim == 1 else a.T
        for a in arr_list
    ]
    return np.concatenate(arr_2d, axis=0)

