import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import collections

k_B = 1.380649e-23  # Constante de Boltzmann (J/K) 

def mod(v):
    return np.sum(v * v, axis=-1)

def pmod(v, T, m):
    """Distribución de Maxwell-Boltzmann teórica"""
    return 4 * np.pi * v**2 * np.power(m / (2 * np.pi * k_B * T), 3 / 2) * np.exp(- m * v**2 / (2 * k_B * T))

class Simulation(animation.TimedAnimation):
    def __init__(self, n_particles, mass, rad, T, V, max_time, dt):
        self.PART = n_particles
        self.MASS = mass
        self.RAD = rad
        self.DIAM = 2 * rad
        self.T = T
        self.V0 = V
        self.V = lambda t: V
        self.Vconst = True

        self.L = np.power(self.V0, 1/3)
        self.halfL = self.L / 2
        self.A = 6 * self.L**2

        self.max_time = max_time
        self.dt = dt
        self.Nt = int(max_time / self.dt)

        self.evaluate_properties()

        self.min_v = 0
        self.max_v = self.vmax * 3
        self.dv = self.max_v / 40.0 
        self.Nv = int((self.max_v - self.min_v) / self.dv)

        self.dP = self.dt * 10 
        self.NP = int(max_time / self.dP) + 2

        # Historiales termodinámicos limitados a los últimos 1000 pasos 
        self.hist_len = 1000
        self.t_hist = collections.deque(maxlen=self.hist_len)
        self.T_hist = collections.deque(maxlen=self.hist_len)
        self.K_hist = collections.deque(maxlen=self.hist_len)
        self.U_hist = collections.deque(maxlen=self.hist_len)
        self.E_hist = collections.deque(maxlen=self.hist_len)

        self.init_particles()
        
        # Parámetros de Lennard-Jones 
        self.EPSILON = 33.3 * k_B
        self.SIGMA = 0.296e-9
        self.F, self.U_pot = self.calcular_fuerzas_lj()

        self.init_figures()

        animation.TimedAnimation.__init__(self, self.fig, interval=20, blit=True, repeat=False)

    def evaluate_properties(self):
        self.P = self.PART * k_B * self.T / self.V0
        self.U = 1.5 * self.PART * k_B * self.T
        self.vrms = np.sqrt(3 * k_B * self.T / self.MASS)
        self.vmax = np.sqrt(2 * k_B * self.T / self.MASS)
        self.vmed = np.sqrt(8 * k_B * self.T / (np.pi * self.MASS))

    def init_particles(self):
        n_lados = int(np.round(self.PART**(1/3)))
        espaciado = np.linspace(-self.halfL * 0.8, self.halfL * 0.8, n_lados)
        x, y, z = np.meshgrid(espaciado, espaciado, espaciado)
        
        posiciones_base = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        self.r = posiciones_base[:self.PART].copy()
        
        distancia = (self.halfL * 1.6) / n_lados
        amplitud = 0.40 * distancia # Ruido del 40% para mayor caos
        perturbaciones = np.random.uniform(-amplitud, amplitud, size=(self.PART, 3))
        self.r += perturbaciones

        sigma_v = np.sqrt(k_B * self.T / self.MASS)
        self.v = np.random.normal(loc=0.0, scale=sigma_v, size=(self.PART, 3))
        self.v -= np.mean(self.v, axis=0)

    def calcular_fuerzas_lj(self):
        diff = self.r[:, np.newaxis, :] - self.r[np.newaxis, :, :]
        r_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r_sq, np.inf)
        
        sr2 = (self.SIGMA**2) / r_sq
        sr6 = sr2**3
        sr12 = sr6**2
        
        F_mag_over_r = (24 * self.EPSILON / r_sq) * (2 * sr12 - sr6)
        F_ij = F_mag_over_r[:, :, np.newaxis] * diff
        F_total = np.sum(F_ij, axis=1)

        # Cálculo de Energía Potencial 
        U_ij = 4 * self.EPSILON * (sr12 - sr6)
        U_total = np.sum(U_ij) / 2.0  # Dividir entre 2 para evitar doble conteo
        
        return F_total, U_total

    def update_temp(self, val):
        T_nueva = self.slider_temp.val
        factor_velocidad = np.sqrt(T_nueva / self.T)
        self.v *= factor_velocidad
        self.T = T_nueva

        vs = np.linspace(0, self.max_v, 50)
        nuevos_valores_y = self.PART * pmod(vs, self.T, self.MASS) * self.dv
        self.line_mb.set_ydata(nuevos_valores_y)
        
    def update_box(self, val):
        """Slider Dimensión de la caja """
        L_nueva = self.slider_box.val * 1e-6
        factor_posicion = L_nueva / self.L
        self.r *= factor_posicion
        self.L = L_nueva
        self.halfL = self.L / 2.0
        self.A = 6 * self.L**2
        self.V0 = self.L**3
        box_limits = [-self.halfL, self.halfL]
        self.ax1.set_xlim3d(box_limits); self.ax1.set_ylim3d(box_limits); self.ax1.set_zlim3d(box_limits)

    def init_figures(self):
        self.fig = plt.figure(figsize=(14, 8))
        plt.subplots_adjust(bottom=0.20, hspace=0.4, wspace=0.3)

        self.ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
        self.ax2 = plt.subplot2grid((3, 3), (2, 0)) # Temperatura
        self.ax3 = plt.subplot2grid((3, 3), (2, 1)) # Energías
        self.ax4 = plt.subplot2grid((3, 3), (2, 2)) # Energía Total
        self.ax5 = plt.subplot2grid((3, 3), (0, 2)) # Histograma
        self.ax6 = plt.subplot2grid((3, 3), (1, 2)) # Presión

        # Setup 3D
        box_limits = [-self.halfL, self.halfL]
        self.ax1.set_xlim3d(box_limits); self.ax1.set_ylim3d(box_limits); self.ax1.set_zlim3d(box_limits)
        self.line_3d = self.ax1.plot([], [], [], ls='None', marker='o', ms=3)[0]
        self.line_3d_cm = self.ax1.plot([0], [0], [0], ls='None', marker='x', color='r')[0]

        # Setup Temperatura 
        self.ax2.set_xlabel('Tiempo [ps]')
        self.ax2.set_ylabel('Temp [K]')
        self.line_T = self.ax2.plot([], [], color='red', lw=1.5)[0]

        # Setup Energías 
        self.ax3.set_xlabel('Tiempo [ps]')
        self.ax3.set_ylabel('Energía [J]')
        self.line_K = self.ax3.plot([], [], color='blue', label='K', lw=1)[0]
        self.line_U = self.ax3.plot([], [], color='orange', label='U', lw=1)[0]
        self.ax3.legend(loc='upper right', fontsize=7)

        # Setup Energía Total
        self.ax4.set_xlabel('Tiempo [ps]')
        self.ax4.set_ylabel('E. Total [J]')
        self.line_E = self.ax4.plot([], [], color='green', lw=1.5)[0]

        # Setup Histograma 
        vs = np.linspace(0, self.max_v, 50)
        self.line_mb, = self.ax5.plot(vs, self.PART * pmod(vs, self.T, self.MASS) * self.dv, color='r')
        self.vel_x = np.linspace(self.min_v, self.max_v, self.Nv)
        self.vel_y = np.zeros(self.Nv)
        self.line_vel = self.ax5.plot([], [], color='b', lw=1.5)[0]

        # Setup Presión 
        self.ax6.set_xlabel('Tiempo [ps]')
        self.ax6.set_ylabel('Presión [Pa]')
        self.ax6.set_xlim(0, self.max_time * 1e12)
        self.ex_p = 0.0 
        self.last_P = -1
        self.P_x = np.zeros(self.NP)
        self.P_y = np.zeros(self.NP)
        self.line_p = self.ax6.plot([], [], color='g', lw=1.5)[0]

        # Sliders interactivos 
        ax_temp = plt.axes([0.15, 0.10, 0.65, 0.02])
        ax_box = plt.axes([0.15, 0.04, 0.65, 0.02])
        self.slider_temp = Slider(ax=ax_temp, label='Temp [K]', valmin=10.0, valmax=1000.0, valinit=self.T)
        self.slider_box = Slider(ax=ax_box, label='Lado [μm]', valmin=2.0, valmax=20.0, valinit=self.L * 1e6)
        self.slider_temp.on_changed(self.update_temp)
        self.slider_box.on_changed(self.update_box)

        self._drawn_artists = [
            self.line_3d, self.line_3d_cm,
            self.line_T, self.line_K, self.line_U, self.line_E,
            self.line_vel, self.line_p
        ]

    def _draw_frame(self, t):
        # Integración Velocity Verlet
        self.r += self.v * self.dt + 0.5 * (self.F / self.MASS) * self.dt**2

        # Colisiones de pared elásticas 
        walls = np.nonzero(np.abs(self.r) + self.RAD > self.halfL)
        v_walls_pre = np.abs(self.v[walls]) 
        
        self.v[walls] *= -1
        self.r[walls] = np.sign(self.r[walls]) * (self.halfL - self.RAD)

        F_nueva, U_nueva = self.calcular_fuerzas_lj()
        self.v += 0.5 * ((self.F + F_nueva) / self.MASS) * self.dt
        self.F = F_nueva
        self.U_pot = U_nueva

        # Cálculos Termodinámicos
        K_total = 0.5 * self.MASS * np.sum(self.v**2)
        T_inst = (2.0 * K_total) / (3.0 * self.PART * k_B)
        E_total = K_total + self.U_pot

        t_ps = t * 1e12 # Tiempo en picosegundos 
        self.t_hist.append(t_ps)
        self.T_hist.append(T_inst)
        self.K_hist.append(K_total)
        self.U_hist.append(self.U_pot)
        self.E_hist.append(E_total)

        # --- ACTUALIZAR GRÁFICOS ---
        CM = np.sum(self.r, axis=0) / self.PART

        self.line_3d.set_data(self.r[:, 0], self.r[:, 1])
        self.line_3d.set_3d_properties(self.r[:, 2])
        self.line_3d_cm.set_data([CM[0]], [CM[1]])
        self.line_3d_cm.set_3d_properties([CM[2]])

        # Termodinámica
        self.line_T.set_data(self.t_hist, self.T_hist)
        self.ax2.set_xlim(self.t_hist[0], self.t_hist[-1] + 1e-5)
        self.ax2.set_ylim(min(self.T_hist)*0.9, max(self.T_hist)*1.1 + 1e-5)

        self.line_K.set_data(self.t_hist, self.K_hist)
        self.line_U.set_data(self.t_hist, self.U_hist)
        self.ax3.set_xlim(self.t_hist[0], self.t_hist[-1] + 1e-5)
        min_KU = min(min(self.K_hist), min(self.U_hist))
        max_KU = max(max(self.K_hist), max(self.U_hist))
        self.ax3.set_ylim(min_KU - abs(min_KU)*0.1, max_KU + abs(max_KU)*0.1 + 1e-25)

        self.line_E.set_data(self.t_hist, self.E_hist)
        self.ax4.set_xlim(self.t_hist[0], self.t_hist[-1] + 1e-5)
        self.ax4.set_ylim(min(self.E_hist) - abs(min(self.E_hist))*0.001, max(self.E_hist) + abs(max(self.E_hist))*0.001 + 1e-25)

        # Histograma 
        v_mod = np.sqrt(mod(self.v))
        for k in range(self.Nv):
            self.vel_y[k] = np.count_nonzero((k*self.dv < v_mod) & (v_mod < (k + 1)*self.dv))
        self.line_vel.set_data(self.vel_x, self.vel_y)

        # Presión 
        self.ex_p += 2 * self.MASS * np.sum(v_walls_pre)
        i = int(t / self.dP)
        if i > self.last_P + 1 and i < self.NP:
            self.last_P = i - 1
            self.P_x[self.last_P] = t_ps
            self.P_y[self.last_P] = self.ex_p / (self.dP * self.A)
            self.ex_p = 0.0
            self.line_p.set_data(self.P_x[:i], self.P_y[:i])
            if i > 1:
                self.ax6.set_ylim(0, np.max(self.P_y[:i]) * 1.1)

    def new_frame_seq(self):
        return iter(np.linspace(0, self.max_time, self.Nt))

# ==========================================
# PARÁMETROS FÍSICOS DEL SISTEMA
# ==========================================

PARTICLES = 200 # 
MASS = 3.32e-27 # kg 
RADIUS = 0.148e-9 # m 
TEMPERATURE = 300.0 # K 
L = 10e-6 # 
V_INICIAL = L**3
DT = 1e-10 # 1 picosegundo, 10e-11 también es bastante estable, para 10e-10 ya hay problemas si las dimensiones de la caja se alteran muy rápido. 
T_MAX = 1000 * DT 

ani = Simulation(PARTICLES, MASS, RADIUS, TEMPERATURE, V_INICIAL, T_MAX, DT)
plt.show()


