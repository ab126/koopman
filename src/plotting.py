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

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=30,
        blit=False   # IMPORTANT for Jupyter
    )

    plt.title("Slab + Rigid Rod Animation")

    # prevent garbage collection
    global anim_ref
    anim_ref = ani

    plt.show()

    return ani


