import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LogNorm
from matplotlib.widgets import RadioButtons

a0 = 1.0  # raggio di Bohr


# === Funzioni d'onda corrette ===
def hydrogen_1s(x, y, z, a0=1.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    psi = (1 / np.sqrt(np.pi * a0**3)) * np.exp(-r / a0)
    return psi

def hydrogen_2s(x, y, z, a0=1.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    psi = (1 / (4 * np.sqrt(2 * np.pi * a0**3))) * (2 - r / a0) * np.exp(-r / (2 * a0))
    return psi

def hydrogen_2p_x(x, y, z, a0=1.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r != 0))
    phi = np.arctan2(y, x)
    psi = (1 / (4 * np.sqrt(2 * np.pi * a0**5))) * (r / a0) * np.exp(-r / (2 * a0)) * np.sin(theta) * np.cos(phi)
    return psi

def hydrogen_2p_y(x, y, z, a0=1.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r != 0))
    phi = np.arctan2(y, x)
    psi = (1 / (4 * np.sqrt(2 * np.pi * a0**5))) * (r / a0) * np.exp(-r / (2 * a0)) * np.sin(theta) * np.sin(phi)
    return psi

def hydrogen_2p_z(x, y, z, a0=1.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r != 0))
    psi = (1 / (4 * np.sqrt(2 * np.pi * a0**5))) * (r / a0) * np.exp(-r / (2 * a0)) * np.cos(theta)
    return psi


# === Griglia 3D ===
n = 40
lim = 9.5
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
z = np.linspace(-lim, lim, n)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()


# === Dizionario delle funzioni ===
orbitals = {
    '1s': hydrogen_1s,
    '2s': hydrogen_2s,
    '2p_x': hydrogen_2p_x,
    '2p_y': hydrogen_2p_y,
    '2p_z': hydrogen_2p_z
}


# === Funzione che disegna un orbitale ===
def compute_orbital_data(name):
    psi = orbitals[name](xf, yf, zf, a0)
    prob = np.abs(psi)**2
    threshold = prob.max() * 1e-3
    mask = prob > threshold
    return xf[mask], yf[mask], zf[mask], prob[mask]


# === Inizializzazione grafico ===
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25)  # spazio per i pulsanti

# Dati iniziali (1s)
xplot, yplot, zplot, probplot = compute_orbital_data('1s')
vmax = probplot.max()
vmin = vmax * 1e-4

sc = ax.scatter(
    xplot, yplot, zplot, c=probplot, cmap='viridis', s=3, alpha=0.6,
    norm=LogNorm(vmin=max(probplot.min(), vmin), vmax=vmax)
)

ax.set_title('Hydrogen orbital: 1s')
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.set_zlabel('z (Å)')
ax.set_box_aspect([1, 1, 1])
cbar = plt.colorbar(sc, ax=ax, pad=0.1, label='Probability density (|ψ|²)')


# === Pulsanti radio per selezionare l’orbitale ===
ax_radio = plt.axes([0.05, 0.3, 0.15, 0.35], facecolor='lightgray')
radio = RadioButtons(ax_radio, ('1s', '2s', '2p_x', '2p_y', '2p_z'), active=0)

def update(label):
    global sc
    # Ricalcola i dati
    xplot, yplot, zplot, probplot = compute_orbital_data(label)
    vmax = probplot.max()
    vmin = vmax * 1e-4

    # Aggiorna lo scatter
    sc.remove()
    sc = ax.scatter(
        xplot, yplot, zplot, c=probplot, cmap='viridis', s=3, alpha=0.6,
        norm=LogNorm(vmin=max(probplot.min(), vmin), vmax=vmax)
    )
    ax.set_title(f'Hydrogen orbital: {label}')
    plt.draw()

radio.on_clicked(update)

plt.show()
