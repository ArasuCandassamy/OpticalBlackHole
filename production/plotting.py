import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os


def load_simulation_data(filename):
    """
    Load simulation data from NPZ file.
    
    Args:
        filename: Path to the .npz file
    
    Returns:
        Dictionary containing all simulation data
    """
    print(f"Loading data from {filename}...")
    data = np.load(filename, allow_pickle=True)
    
    # Convert NPZ structure to dictionary
    result = {}
    for key in data.files:
        result[key] = data[key]
    
    print("Data loaded successfully!")
    return result


def plot_simulation(data, save=True):
    """
    Generate visualization from loaded simulation data.
    
    Args:
        data: Dictionary containing simulation data
        save: If True, save the figure with parameters in the filename
    """
    # Extract parameters
    params = data['parameters'].item()
    sim_grid = data['simulation_grid'].item()
    fields = data['fields'].item()
    device = data['device'].item()
    trajectories = data['trajectories'].item()
    
    # Unpack variables
    L = params['L']
    b_inf = params['b_inf']
    a_hat = params['a_hat']
    Q_hat = params['Q_hat']
    R_h = params['R_h']
    LAMBDA_0 = params['LAMBDA_0']
    resolution = params['resolution']
    
    X = sim_grid['X']
    Y = sim_grid['Y']
    R_grid = sim_grid['R_grid']
    
    Ez = fields['Ez']
    Sx = fields['Sx']
    Sy = fields['Sy']
    
    r_boundaries = device['r_boundaries']
    
    X_geo = trajectories['X_geo']
    Y_geo = trajectories['Y_geo']
    X_eff = trajectories['X_eff']
    Y_eff = trajectories['Y_eff']
    
    # Create figure
    extent = [-L/2, L/2, -L/2, L/2]
    fig, ax = plt.subplots(figsize=(7, 6))
    mag = np.abs(Ez)
    mag_normalized = mag / np.max(mag)
    
    im2 = ax.imshow(mag_normalized.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax, label='|E| (norm)')
    
    # Drawing the annuli
    theta_center = 225; theta_width = 18
    theta1 = theta_center - theta_width/2; theta2 = theta_center + theta_width/2
    for i, r_b in enumerate(r_boundaries):
        if i == 0 or i == len(r_boundaries) - 1:
            lw = 0.8 if i == len(r_boundaries) - 1 else 0.5
            alpha = 0.6 if i == len(r_boundaries) - 1 else 0.4
            ax.add_patch(Circle((0,0), r_b, color='white', fill=False, ls='-', linewidth=lw, alpha=alpha))
        else:
            ax.add_patch(Arc((0,0), 2*r_b, 2*r_b, angle=0, theta1=theta1, theta2=theta2, 
                                 color='white', ls='-', linewidth=0.5, alpha=0.5))
    
    # --- POYNTING QUIVER ---
    step = 30
    sl_x = slice(0, None, step); sl_y = slice(None, None, step)
    R_safe = R_grid.copy()
    R_sub = R_safe[sl_x, sl_y]
    R_sub[R_sub < 0.1] = 0.1 
    U_raw = Sx[sl_x, sl_y] / R_sub; V_raw = Sy[sl_x, sl_y] / R_sub
    
    X_sub = X[sl_x, sl_y]; Y_sub = Y[sl_x, sl_y]
    R_limit_draw = r_boundaries[-1] 
    mask_in = np.sqrt(X_sub**2 + Y_sub**2) <= R_limit_draw
    
    U_in = U_raw[mask_in]; V_in = V_raw[mask_in]
    X_in = X_sub[mask_in]; Y_in = Y_sub[mask_in]
    
    max_val = np.max(np.hypot(U_in, V_in)) if len(U_in) > 0 else 1.0
    ax.quiver(X_in, Y_in, U_in/max_val, V_in/max_val, color='white', pivot='tail', scale=10)
    
    # --- TRAJECTORY PLOTS ---
    # 1. Geodesic (white dashed)
    ax.plot(X_geo, Y_geo, color='white', linewidth=5, linestyle='--', alpha=0.8, label='Geodesic (Theory)')
    
    # 2. Effective Poynting (solid red)
    ax.plot(X_eff, Y_eff, color='red', linewidth=2.0, alpha=1.0, label='Poynting Avg (Sim)')
    
    # Decorative elements
    ax.add_patch(Circle((0,0), R_h, color='white', fill=False, ls='-', linewidth=2))
    ax.text(-R_h + 0.5, 0, r'$R_h$', color='white', fontsize=12)
    
    label_text = r'$\hat{b}_{\infty} = ' + f'{b_inf}' + r'$' if (a_hat==0 and Q_hat==0) else r'$\hat{a}=' + f'{a_hat}' + r', \rho_Q=' + f'{Q_hat}' + r'$'
    ax.text(-L/2 + 1, -L/2 + 1, label_text, color='white', fontsize=14)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.5, facecolor='black', edgecolor='white', labelcolor='white')
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure with parameters in filename
    if save:
        fig_filename = f'fdfd_plot_a{a_hat}_Q{Q_hat}_b{b_inf}_res{resolution}.pdf'
        plt.savefig(fig_filename, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {fig_filename}")
    
    plt.show()


def find_latest_data_file():
    """
    Find the most recent .npz file in the current directory.
    
    Returns:
        Path to the most recent .npz file, or None if not found
    """
    files = glob.glob('fdfd_data_*.npz')
    if not files:
        return None
    return max(files, key=os.path.getctime)


if __name__ == "__main__":
    # Try to find and load the latest data file
    print("List of available data files:")
    files = glob.glob('fdfd_data_*.npz')
    files = sorted(files)
    for i, f in enumerate(files, start=1):
        print(f"{i}: {f}")
    
    print("Index of plotting file :")
    index = int(input("Enter the index of the file to plot (0 for latest, -1 for all): "))
    if index == 0:
        data_file = find_latest_data_file()
    else:
        if index == -1:
            data_file = files  
        else:
            if 0 < index <= len(files):
                data_file = files[index - 1]
            else:
                data_file = None
    
    if data_file is None:
        print("No data file found. Please run FDFD_simulation.py first.")
        print("Usage: python FDFD_simulation.py")
    else:
        for data_file in files if index == -1 else [data_file]:
            data = load_simulation_data(data_file)
            plot_simulation(data, save=True)