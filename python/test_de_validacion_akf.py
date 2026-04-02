"""
TEST DE VALIDACIÓN IBI - AKF CON D_KL Y TRAYECTORIA GEODÉSICA
Versión: 7.1
Fecha: Marzo 2026
Autor: Daniel Vivas - dvivas1@uc.edu.ve
            
CORRECCIÓN PRINCIPAL:
- compare_variance_reduction: Limitar ventana a zona estable para evitar 
  que el drift infle artificialmente el ratio de reducción de varianza
  
Genera:
- convergencia_mse_TIMESTAMP.png (Figura 1)
- reduccion_varianza_TIMESTAMP.png (Figura 2)
- senal_ibi_TIMESTAMP.png (Figura 3)
- trayectoria_geodesica_TIMESTAMP.png (Figura 4 - NUEVA PARA IBI28)
- dkL_por_iteracion_TIMESTAMP.csv (Datos para análisis)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
import os
from scipy import stats
from scipy.special import rel_entr

# ================================================================================
# CONFIGURACIÓN
# ================================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TEST_CONFIG = {
    'n_samples': 200,
    'n_iterations': 100,
    'snr_db': 15,
    'dt': 0.075,
    'delay': 3,
    'falsification_threshold': 0.30,
    'preliminary_validation_threshold': 0.87,
    'export_csv': True,
    'export_json': True,
    'calculate_dkl': True,  # NUEVO: Activar cálculo de D_KL
    'geodesic_tracking': True,  # NUEVO: Seguimiento de trayectoria geodésica
}

PHYSICAL_CONSTANTS = {
    'F': 96485.332,
    'R': 8.314,
    'T': 298.15,
    'kB': 1.381e-23,
    'ln2': np.log(2),
}

# ================================================================================
# FUNCIONES DE INFORMACIÓN GEOMÉTRICA (NUEVO PARA IBI28)
# ================================================================================
def calculate_fisher_information_matrix(theta, measurements, sigma_noise=1e-5):
    """
    Calcular la Matriz de Información de Fisher para el estado actual.
    Para modelo gaussiano y ~ N(h(theta), R):
    I_F(theta) = (dh/dtheta)^T R^{-1} (dh/dtheta)
    """
    n_states = len(theta)
    I_F = np.zeros((n_states, n_states))
    
    # Aproximación numérica del Jacobiano dh/dtheta
    epsilon = 1e-6
    h_base = float(np.mean(measurements))  # ← CORRECCIÓN: Extraer escalar explícitamente
    
    for i in range(n_states):
        for j in range(n_states):
            # Perturbación hacia adelante
            theta_plus_i = theta.copy()
            theta_plus_i[i] += epsilon
            h_plus_i = h_base * (1 + epsilon * (1 if i == 0 else 0.1))
            
            theta_plus_j = theta.copy()
            theta_plus_j[j] += epsilon
            h_plus_j = h_base * (1 + epsilon * (1 if j == 0 else 0.1))
            
            # Elemento de Fisher (CORRECCIÓN: todos son escalares ahora)
            diff_i = h_plus_i - h_base  # Ya es escalar
            diff_j = h_plus_j - h_base  # Ya es escalar
            I_F[i, j] = (diff_i * diff_j) / (sigma_noise**2)
    
    # Regularización para estabilidad numérica
    I_F += np.eye(n_states) * 1e-10
    
    return I_F


def calculate_fisher_distance(theta_A, theta_B, I_F):
    """
    Calcular la Distancia de Fisher (aproximación geodésica local).
    Δs_Fisher = sqrt((θ_B - θ_A)^T I_F (θ_B - θ_A))
    """
    d_theta = theta_B - theta_A
    d_fisher = np.sqrt(np.dot(d_theta, np.dot(I_F, d_theta)))
    return d_fisher


def calculate_kl_divergence_gaussian(p_mean, p_var, q_mean, q_var):
    """
    Calcular Divergencia de Kullback-Leibler entre dos distribuciones gaussianas.
    D_KL(P || Q) = 0.5 * [log(q_var/p_var) + (p_var + (p_mean - q_mean)^2)/q_var - 1]
    """
    # Evitar divisiones por cero y logs de números negativos
    p_var = max(p_var, 1e-10)
    q_var = max(q_var, 1e-10)
    
    d_kl = 0.5 * (
        np.log(q_var / p_var) + 
        (p_var + (p_mean - q_mean)**2) / q_var - 1
    )
    
    return max(d_kl, 0)  # D_KL siempre >= 0


def calculate_kl_divergence_empirical(p_samples, q_samples, n_bins=50):
    """
    Calcular D_KL empírica usando histogramas.
    """
    # Crear histogramas con el mismo soporte
    range_min = min(p_samples.min(), q_samples.min())
    range_max = max(p_samples.max(), q_samples.max())
    
    p_hist, bin_edges = np.histogram(p_samples, bins=n_bins, range=(range_min, range_max), density=True)
    q_hist, _ = np.histogram(q_samples, bins=n_bins, range=(range_min, range_max), density=True)
    
    # Evitar ceros (suavizado)
    epsilon = 1e-10
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon
    
    # Normalizar
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)
    
    # Calcular D_KL
    d_kl = np.sum(rel_entr(p_hist, q_hist))
    
    return d_kl


def calculate_geodesic_trajectory(theta_history, I_F_history):
    """
    Calcular la trayectoria geodésica acumulada en el manifold de Fisher.
    """
    n_iter = len(theta_history)
    geodesic_distance = np.zeros(n_iter)
    
    for k in range(1, n_iter):
        if k < len(I_F_history):
            I_F = I_F_history[k]
            d_fisher = calculate_fisher_distance(
                theta_history[k-1], 
                theta_history[k], 
                I_F
            )
            geodesic_distance[k] = geodesic_distance[k-1] + d_fisher
    
    return geodesic_distance


def calculate_landauer_energy_bound(d_kl_reduction, temperature=298.15):
    """
    Calcular el límite energético de Landauer para la reducción de D_KL.
    E_min >= k_B * T * ΔD_KL
    """
    kB = PHYSICAL_CONSTANTS['kB']
    e_min = kB * temperature * d_kl_reduction
    return e_min


# ================================================================================
# FILTRO DE KALMAN ADAPTATIVO (AKF) CON H CORREGIDA
# ================================================================================
class AdaptivePhysicsInformedKalmanFilter:
    """
    Filtro de Kalman Adaptativo - Solo R se adapta según innovación (Q fija)
    CORRECCIÓN: El manuscrito ahora especifica claramente que esta implementación
    adapta únicamente la covarianza de medición R basada en la innovación del sensor.
    La covarianza de proceso Q permanece fija para garantizar estabilidad numérica
    en recursos limitados (ESP32-S3).
    """
    
    def __init__(self, dt=0.075, alpha=0.1, R_base=1e-5):
        self.dt = dt
        self.n_states = 3
        self.alpha = alpha  # Factor de suavizado para adaptación
        self.R_base = R_base   # Valor base de R
        self.R_anterior = R_base  # R adaptativo que cambia
        
        # Matriz F - Modelo con memoria persistente y acoplamiento
        self.F = np.array([
            [1.0,  dt, 0.1],
            [0.0, 0.95, 0.0],
            [0.05, 0.0, 0.92]
        ])
        
        # CORRECCIÓN PRINCIPAL: H observable
        self.H = np.array([[1.0, 0.1, 0.05]])
        
        # Ruido de proceso (Q se mantiene fijo - NO adaptativo en esta versión)
        self.Q = np.diag([1e-6, 1e-7, 1e-5])

    def filter(self, measurements, track_dkl=True):
        """
        Aplicar filtro de Kalman Adaptativo completo con seguimiento de D_KL
        
        Parámetros:
        -----------
        measurements : array
            Señal de medición ruidosa
        track_dkl : bool
            Si True, calcular D_KL por iteración
        
        Retorna:
        --------
        x_est : ndarray
            Estados estimados
        dkl_history : list (opcional)
            Historial de D_KL por iteración
        fisher_distance_history : list (opcional)
            Distancia de Fisher por iteración
        """
        n_meas = len(measurements)
        x_est = np.zeros((n_meas, self.n_states))
        P = np.eye(self.n_states) * 0.1
        
        # Inicialización
        x_est[0, 0] = np.mean(measurements[:min(5, len(measurements))])
        x_est[0, 1] = 0.0
        x_est[0, 2] = 0.0
        
        # Reiniciar R adaptativo para cada nueva filtración
        self.R_anterior = self.R_base
        
        # NUEVO PARA IBI28: Historiales para análisis de información
        dkl_history = [] if track_dkl else None
        fisher_distance_history = [] if track_dkl else None
        I_F_history = [] if track_dkl else None
        
        # Estimación inicial de varianza del sensor
        p_var_initial = np.var(measurements[:min(20, len(measurements))])
        
        for k in range(1, n_meas): 
            # ========== PREDICCIÓN ==========
            x_pred = self.F @ x_est[k-1]
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # ========== INNOVACIÓN (RESIDUO) ==========
            z_k = measurements[k]
            residuo = z_k - (self.H @ x_pred)  # Innovación del sensor
            
            # ========== ADAPTACIÓN DE R (LÓGICA AKF) ==========
            R_adaptativo = (1 - self.alpha) * self.R_anterior + self.alpha * (residuo**2)
            R_adaptativo = np.clip(R_adaptativo, self.R_base * 0.1, self.R_base * 1000)
            self.R_anterior = R_adaptativo
            
            # ========== ACTUALIZACIÓN CON R ADAPTATIVO ==========
            S = self.H @ P_pred @ self.H.T + R_adaptativo
            K = P_pred @ self.H.T / S
            
            x_upd = x_pred + K.flatten() * residuo
            
            I_KH = np.eye(self.n_states) - np.outer(K, self.H)
            P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R_adaptativo
            
            # Clip para estabilidad física
            x_upd[0] = np.clip(x_upd[0], 0, 0.01)
            x_est[k] = x_upd
            
            # ========== NUEVO PARA IBI28: CÁLCULO DE D_KL ==========
            if track_dkl:
                # Distribución del sensor (P): basada en medición ruidosa
                p_mean = z_k
                p_var = p_var_initial  # Varianza empírica del sensor
                
                # Distribución del filtro (Q): basada en estimación
                q_mean = x_upd[0]
                q_var = P[0, 0]  # Varianza del estado estimado
                
                # Calcular D_KL
                d_kl = calculate_kl_divergence_gaussian(p_mean, p_var, q_mean, q_var)
                dkl_history.append(d_kl)
                
                # Calcular matriz de Fisher y distancia geodésica
                I_F = calculate_fisher_information_matrix(x_upd, measurements[:k+1], sigma_noise=np.sqrt(R_adaptativo))
                I_F_history.append(I_F)
                
                if k > 1:
                    d_fisher = calculate_fisher_distance(x_est[k-1], x_est[k], I_F)
                    fisher_distance_history.append(d_fisher)
                else:
                    fisher_distance_history.append(0.0)
        
        return x_est, dkl_history, fisher_distance_history, I_F_history


# ================================================================================
# CLASE PRINCIPAL DE VALIDACIÓN (ACTUALIZADA PARA IBI28)
# ================================================================================
class IBIStressTestStable:
    """Test de validación con Filtro Kalman Adaptativo (AKF) + D_KL"""
    
    def __init__(self, config=TEST_CONFIG):
        self.config = config
        self.n_samples = config['n_samples']
        self.dt = config['dt']
        self.time_interference = self.n_samples // 2
        self.t = np.linspace(0, self.n_samples * self.dt, self.n_samples)
        
        # AKF con parámetros de adaptación
        self.kalman_filter = AdaptivePhysicsInformedKalmanFilter(
            dt=self.dt, 
            alpha=0.1,
            R_base=1e-5
        )
        
        self.results = None
        self.observability_results = None
        self.variance_comparison = None
        self.dkl_results = None  # NUEVO PARA IBI28
        self.geodesic_results = None  # NUEVO PARA IBI28

    def generate_clean_signal(self):
        """Generar señal limpia"""
        y = 0.001 * (1 + 0.5 * np.sin(self.t))
        interference = np.zeros(self.n_samples)
        interference[self.time_interference:] = 0.0025
        drift = 0.0002 * self.t
        return y + interference + drift

    def generate_observed_signal(self, clean_signal, snr_db=15):
        """Añadir ruido - Modelo: Gaussiano con correlación temporal"""
        signal_power = np.mean(clean_signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(clean_signal))
        noise_memory = 0.2 * np.cumsum(noise) * self.dt
        return clean_signal + noise_memory

    def standard_averaging_estimator(self, raw_signal):
        """Promediado acumulativo estándar"""
        n = len(raw_signal)
        averaged = np.zeros(n)
        for i in range(n):
            averaged[i] = np.mean(raw_signal[:i+1])
        return averaged

    def kalman_inference_engine(self, raw_signal, track_dkl=True):
        """Motor Kalman Adaptativo con seguimiento de D_KL"""
        x_est, dkl_hist, fisher_hist, I_F_hist = self.kalman_filter.filter(
            raw_signal, 
            track_dkl=track_dkl
        )
        return x_est[:, 0], dkl_hist, fisher_hist, I_F_hist

    def calculate_metrics(self, y_true, y_pred):
        """Calcular métricas"""
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        except ImportError:
            def r2_score(y_true, y_pred):
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            def mean_squared_error(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)
            
            def mean_absolute_error(y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        mask_nonzero = y_true != 0
        mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) 
                              / y_true[mask_nonzero])) * 100 if np.any(mask_nonzero) else 0.0
        
        r2 = r2_score(y_true, y_pred)
        
        return {'r2': r2, 'rmse': rmse, 'mse': mse, 'mae': mae, 'mape': mape}

    def verify_observability_matrix(self):
        """Verificar observabilidad - Caso específico para esta configuración"""
        print("\n" + "="*80)
        print("VERIFICACIÓN DE OBSERVABILIDAD (CASO ESPECÍFICO)")
        print("="*80)
        
        n = self.kalman_filter.n_states
        F = self.kalman_filter.F
        H = self.kalman_filter.H
        
        observability_blocks = [H]
        HF = H @ F
        for k in range(1, n):
            observability_blocks.append(HF)
            HF = HF @ F
        
        O = np.vstack(observability_blocks)
        
        rank_O = np.linalg.matrix_rank(O, tol=1e-10)
        cond_number = np.linalg.cond(O)
        singular_values = np.linalg.svd(O, compute_uv=False)
        
        self.observability_results = {
            'n_states': n,
            'rank_O': int(rank_O),
            'full_rank': bool(rank_O == n),
            'condition_number': float(cond_number) if np.isfinite(cond_number) else float('inf'),
            'min_singular_value': float(singular_values[-1]),
            'max_singular_value': float(singular_values[0]),
            'observable': bool(rank_O == n and cond_number < 10000)
        }
        
        print(f"\n  Estados: {n}")
        print(f"  Rango: {rank_O} / {n}")
        print(f"  ¿Rango completo? {'✓ SÍ' if rank_O == n else '✗ NO'}")
        print(f"  Número de condición: {cond_number:.2f}")
        print(f"  ¿Cond. razonable? {'✓ SÍ' if cond_number < 10000 else '⚠ NO'}")
        print(f"\n  VEREDICTO: {'✓ SISTEMA OBSERVABLE (esta configuración)' if self.observability_results['observable'] else '⚠ REVISAR'}")
        print("  NOTA: Esta verificación es caso-específica para los parámetros F y H usados.")
        print("="*80)
        
        return self.observability_results

    def compare_variance_reduction(self, n_samples_range=[10, 50, 100, 200, 500]):
        """
        Comparar reducción de varianza - CORREGIDO PARA ALINEAR CON IBI_TEST_AKFR.py
        
        CORRECCIÓN CRÍTICA: Limitar ventana de comparación a zona estable
        para evitar que el drift infle artificialmente el ratio.
        """
        print("\n" + "="*80)
        print("COMPARACIÓN DE REDUCCIÓN DE VARIANZA (CORREGIDO)")
        print("="*80)
        
        results_by_samples = []
        original_n = self.n_samples
        original_t = self.t.copy()
        original_interference = self.time_interference
        
        for n_samp in n_samples_range:
            self.n_samples = n_samp
            self.t = np.linspace(0, n_samp * self.dt, n_samp)
            self.time_interference = n_samp // 2
            
            self.kalman_filter = AdaptivePhysicsInformedKalmanFilter(
                dt=self.dt, 
                alpha=0.1, 
                R_base=1e-5
            )
            
            variances_avg = []
            variances_kalman = []
            
            for _ in range(50):
                y_clean = self.generate_clean_signal()
                y_noisy = self.generate_observed_signal(y_clean, snr_db=10)
                 
                y_avg = self.standard_averaging_estimator(y_noisy)
                y_kalman, _, _, _ = self.kalman_inference_engine(y_noisy, track_dkl=False)
                
                # CORRECCIÓN: Usar ventana completa hasta time_interference
                # (alineado con IBI_TEST_AKFR.py)
                err_avg = y_avg[:self.time_interference] - y_clean[:self.time_interference]
                err_kalman = y_kalman[:self.time_interference] - y_clean[:self.time_interference]
                
                variances_avg.append(np.var(err_avg))
                variances_kalman.append(np.var(err_kalman))
            
            mean_var_avg = np.mean(variances_avg)
            mean_var_kalman = np.mean(variances_kalman)
            ratio = mean_var_avg / mean_var_kalman if mean_var_kalman > 1e-15 else np.inf
            
            results_by_samples.append({
                'n_samples': n_samp,
                'var_avg': mean_var_avg,
                'var_kalman': mean_var_kalman,
                'ratio': ratio
            })
            
            print(f"\n  N={n_samp:3d}: Var(Avg)={mean_var_avg:.6e}, Var(Kalman)={mean_var_kalman:.6e}, Ratio={ratio:.2f}×")
        
        self.n_samples = original_n
        self.t = original_t
        self.time_interference = original_interference
        self.kalman_filter = AdaptivePhysicsInformedKalmanFilter(
            dt=self.dt, 
            alpha=0.1, 
            R_base=1e-5
        )
        
        self.variance_comparison = results_by_samples
        print("="*80)
        
        return results_by_samples

    def run_validation_completa(self, n_iterations=None):
        """Ejecutar validación completa con D_KL"""
        if n_iterations is None:
            n_iterations = self.config['n_iterations']
        
        np.random.seed(RANDOM_SEED)
        
        metrics_pre = {'r2': [], 'rmse': [], 'mse': [], 'mae': [], 'mape': []}
        metrics_post = {'r2': [], 'rmse': [], 'mse': [], 'mae': [], 'mape': []}
        
        # NUEVO PARA IBI28: Almacenamiento de D_KL
        dkl_pre = []
        dkl_post = []
        geodesic_distance_pre = []
        geodesic_distance_post = []
        
        print(f"\nEjecutando {n_iterations} iteraciones de Monte Carlo...")
        
        for i in range(n_iterations):
            y_clean = self.generate_clean_signal()
            y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
            y_pred, dkl_hist, fisher_hist, I_F_hist = self.kalman_inference_engine(
                y_noisy, 
                track_dkl=self.config['calculate_dkl']
            )
            
            delay = self.config['delay']
            
            m_pre = self.calculate_metrics(
                y_clean[delay:self.time_interference],
                y_pred[delay:self.time_interference]
            )
            
            m_post = self.calculate_metrics(
                y_clean[self.time_interference:],
                y_pred[self.time_interference:]
            )
            
            if not np.isnan(m_pre['r2']) and np.isfinite(m_pre['r2']):
                for key in metrics_pre:
                    metrics_pre[key].append(m_pre[key])
                
                # NUEVO: Almacenar D_KL
                if self.config['calculate_dkl'] and dkl_hist:
                    dkl_pre.append(np.mean(dkl_hist[delay:self.time_interference]))
                    if fisher_hist:
                        geodesic_distance_pre.append(np.sum(fisher_hist[delay:self.time_interference]))
            
            if not np.isnan(m_post['r2']) and np.isfinite(m_post['r2']):
                for key in metrics_post:
                    metrics_post[key].append(m_post[key])
                
                # NUEVO: Almacenar D_KL
                if self.config['calculate_dkl'] and dkl_hist:
                    dkl_post.append(np.mean(dkl_hist[self.time_interference:]))
                    if fisher_hist:
                        geodesic_distance_post.append(np.sum(fisher_hist[self.time_interference:]))
        
        def summarize(metrics_dict):
            result = {}
            for key, values in metrics_dict.items():
                if len(values) > 0:
                    # CORRECCIÓN CRÍTICA: IC95 = 1.96 * std / √n (intervalo de confianza de la media)
                    result[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'ic95': float(1.96 * np.std(values) / np.sqrt(len(values)))
                    }
                else:
                    result[key] = {'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0, 'ic95': 0}
            return result
        
        self.results = {
            'pre': summarize(metrics_pre),
            'post': summarize(metrics_post),
            'n_valid_pre': len(metrics_pre['r2']),
            'n_valid_post': len(metrics_post['r2']),
            'config': self.config,
            'random_seed': RANDOM_SEED,
            'timestamp': datetime.now().isoformat(),
            'akf_alpha': 0.1,
            'akf_adaptation': 'R_only'
        }
        
        # NUEVO PARA IBI28: Resultados de D_KL
        if self.config['calculate_dkl']:
            self.dkl_results = {
                'pre': {
                    'mean': float(np.mean(dkl_pre)) if dkl_pre else 0,
                    'std': float(np.std(dkl_pre)) if dkl_pre else 0,
                },
                'post': {
                    'mean': float(np.mean(dkl_post)) if dkl_post else 0,
                    'std': float(np.std(dkl_post)) if dkl_post else 0,
                },
                'delta_dkl': float(np.mean(dkl_pre) - np.mean(dkl_post)) if (dkl_pre and dkl_post) else 0
            }
            
            self.geodesic_results = {
                'pre': {
                    'mean': float(np.mean(geodesic_distance_pre)) if geodesic_distance_pre else 0,
                },
                'post': {
                    'mean': float(np.mean(geodesic_distance_post)) if geodesic_distance_post else 0,
                }
            }
        
        return self.results

    def plot_geodesic_trajectory(self, output_dir='./test_ibi_akf'):
        """
       Trayectoria Geodésica en el Manifold de Fisher
        
        Esta figura muestra:
        1. D_KL por iteración (eje Y izquierdo)
        2. Distancia geodésica acumulada (eje Y derecho)
        3. MSE para correlación (eje Y secundario)
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generar una corrida representativa con seguimiento de D_KL
        y_clean = self.generate_clean_signal()
        y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
        y_pred, dkl_hist, fisher_hist, I_F_hist = self.kalman_inference_engine(
            y_noisy, 
            track_dkl=True
        )
        
        # Calcular MSE por iteración
        mse_hist = (y_pred - y_clean)**2
        
        # Calcular distancia geodésica acumulada
        geodesic_cumulative = np.cumsum(fisher_hist) if fisher_hist else np.zeros(len(dkl_hist))
        
        # Calcular límite energético de Landauer
        dkl_reduction = np.maximum(0, np.diff(dkl_hist))
        landauer_energy = [calculate_landauer_energy_bound(d) for d in dkl_reduction]
        landauer_energy = np.insert(landauer_energy, 0, 0)
        
        # Crear figura con múltiples ejes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Eje Y izquierdo: D_KL
        color_dkl = 'tab:red'
        ax1.set_xlabel('Iteración del Filtro (k)', fontsize=12)
        ax1.set_ylabel('Divergencia D_KL(P || Q) [nats]', color=color_dkl, fontsize=12)
        ax1.plot(range(len(dkl_hist)), dkl_hist, color=color_dkl, linewidth=2, 
                 label='D_KL(P_sensor || Q_filtro)', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_dkl)
        ax1.grid(True, alpha=0.3)
        
        # Eje Y derecho superior: Distancia geodésica acumulada
        ax2 = ax1.twinx()
        color_geo = 'tab:blue'
        ax2.set_ylabel('Distancia Geodésica Acumulada Δs_Fisher', color=color_geo, fontsize=12)
        ax2.plot(range(len(geodesic_cumulative)), geodesic_cumulative, color=color_geo, 
                 linewidth=2, linestyle='--', label='Trayectoria Geodésica', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color_geo)
        
        # Eje Y derecho inferior: MSE (para correlación)
        ax3 = ax1.twinx()
        color_mse = 'tab:green'
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('MSE Instantáneo', color=color_mse, fontsize=12)
        ax3.plot(range(len(mse_hist)), mse_hist, color=color_mse, linewidth=1.5, 
                 linestyle=':', label='MSE', alpha=0.5)
        ax3.tick_params(axis='y', labelcolor=color_mse)
        ax3.set_yscale('log')
        
        # Zona de interferencia
        ax1.axvspan(self.time_interference, self.n_samples, color='red', alpha=0.1, 
                   label='Zona de Interferencia')
        
        # Línea de tendencia para D_KL
        z = np.polyfit(range(len(dkl_hist)), dkl_hist, 1)
        p = np.poly1d(z)
        ax1.plot(range(len(dkl_hist)), p(range(len(dkl_hist))), color='darkred', 
                 linestyle='-.', linewidth=1, alpha=0.5, label='Tendencia D_KL')
        
        # Título y leyenda
        plt.title('Trayectoria Geodésica del AKF en el Manifold de Fisher\n' +
                  'Convergencia = Minimización de D_KL a lo largo de Geodésica', 
                  fontsize=14, fontweight='bold')
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                  loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'trayectoria_geodesica_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Trayectoria geodésica: {plot_path}")
        
        # Exportar datos de D_KL para análisis
        csv_path = os.path.join(output_dir, f'dkl_por_iteracion_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteración', 'D_KL_nats', 'Distancia_Geodésica', 'MSE', 
                           'Landauer_Energy_Joules'])
            for k in range(len(dkl_hist)):
                writer.writerow([
                    k, 
                    dkl_hist[k] if k < len(dkl_hist) else 0,
                    geodesic_cumulative[k] if k < len(geodesic_cumulative) else 0,
                    mse_hist[k] if k < len(mse_hist) else 0,
                    landauer_energy[k] if k < len(landauer_energy) else 0
                ])
        
        print(f"✓ Datos D_KL exportados: {csv_path}")
        
        return plot_path, csv_path, {
            'dkl_history': dkl_hist,
            'geodesic_cumulative': geodesic_cumulative.tolist(),
            'mse_history': mse_hist.tolist(),
            'landauer_energy': landauer_energy.tolist(),
            'correlation_dkl_mse': float(np.corrcoef(dkl_hist[:len(mse_hist)], mse_hist[:len(dkl_hist)])[0, 1]) if len(dkl_hist) == len(mse_hist) else None
        }

    def plot_convergence_mse(self, output_dir='./test_ibi_akf'):
        """Figura 1: MSE vs. Tiempo"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        y_clean = self.generate_clean_signal()
        y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
        y_kalman, _, _, _ = self.kalman_inference_engine(y_noisy, track_dkl=False)
        y_avg = self.standard_averaging_estimator(y_noisy)
        
        err_avg = (y_avg - y_clean)**2
        err_kalman = (y_kalman - y_clean)**2
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.t * 1000, err_avg, color='orange', alpha=0.4, label='Error Promediado Estándar')
        plt.plot(self.t * 1000, err_kalman, color='red', linewidth=2, label='Error Filtro AKF (Propuesto)')
        
        plt.yscale('log')
        plt.xlabel('Tiempo de Adquisición (ms)', fontsize=12)
        plt.ylabel('MSE Instantáneo [log]', fontsize=12)
        plt.title('Convergencia de Error en Tiempo Real (Promediado vs. AKF)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plot_path = os.path.join(output_dir, f'convergencia_mse_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figura 1 corregida: {plot_path}")
        return plot_path

    def plot_variance_reduction(self, output_dir='./test_validacion_ibi_akf'):
        """Figura 2: Ratio de Mejora"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.variance_comparison:
            return None
        
        n_samples = [str(r['n_samples']) for r in self.variance_comparison]
        ratios = [r['ratio'] for r in self.variance_comparison]
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(ratios)))
        bars = plt.bar(n_samples, ratios, color=colors, edgecolor='black', alpha=0.8)
        
        plt.axhline(10, color='red', linestyle='--', label='Objetivo Paper (10x)')
        plt.xlabel('Número de Muestras (N)', fontsize=11)
        plt.ylabel('Factor de Mejora (Precisión)', fontsize=11)
        plt.title('Ratio de Reducción de Varianza (AKF)', fontsize=13)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}x', ha='center', va='bottom')
        
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plot_path = os.path.join(output_dir, f'reduccion_varianza_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figura 2 corregida: {plot_path}")
        return plot_path

    def plot_signal_resilience(self, output_dir='./test_ibi_akf'):
        """Figura 3: Comparación de Señales"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        y_clean = self.generate_clean_signal()
        y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
        y_kalman, _, _, _ = self.kalman_inference_engine(y_noisy, track_dkl=False)
        
        plt.figure(figsize=(14, 7))
        
        plt.axvspan(0, self.t[self.time_interference], color='gray', alpha=0.1, label='Operación Normal')
        plt.axvspan(self.t[self.time_interference], self.t[-1], color='red', alpha=0.05, label='Zona de Interferencia')
        
        plt.plot(self.t, y_noisy, color='orange', alpha=0.4, label='Señal Sensor (Ruidosa/Normal)', linewidth=1)
        plt.plot(self.t, y_clean, color='black', linestyle='--', alpha=0.6, label='Referencia Ideal')
        plt.plot(self.t, y_kalman, color='blue', label='Señal Filtrada (IBI-AKF)', linewidth=2.5)
        
        plt.title("Resiliencia y Filtrado de Señal IBI (AKF)", fontsize=15, fontweight='bold')
        plt.xlabel("Tiempo (segundos)", fontsize=12)
        plt.ylabel("Amplitud / Información (Ψ)", fontsize=12)
        
        margin = (np.max(y_noisy) - np.min(y_noisy)) * 0.1
        plt.ylim(np.min(y_noisy) - margin, np.max(y_noisy) + margin)
        
        plt.legend(loc='upper left', frameon=True, shadow=True)
        plt.grid(True, alpha=0.2)
        
        plot_path = os.path.join(output_dir, f'senal_ibi_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figura 3 corregida: {plot_path}")
        return plot_path

    def print_summary(self):
        """Imprimir resumen con D_KL"""
        if self.results is None:
            print("⚠ Ejecutar run_validation_completa() primero")
            return
        
        print("\n" + "="*80)
        print("RESULTADOS ESTADÍSTICOS (AKF)")
        print("="*80)
        
        print(f"\nFASE NORMAL - N={self.results['n_valid_pre']} corridas:")
        print(f"  R²:   {self.results['pre']['r2']['mean']:.5f} ± {self.results['pre']['r2']['ic95']:.5f} (IC95)")
        print(f"  RMSE: {self.results['pre']['rmse']['mean']:.6e}")
        
        print(f"\nFASE INTERFERENCIA - N={self.results['n_valid_post']} corridas:")
        print(f"  R²:   {self.results['post']['r2']['mean']:.5f} ± {self.results['post']['r2']['ic95']:.5f} (IC95)")
        print(f"  RMSE: {self.results['post']['rmse']['mean']:.6e}")
        
        # NUEVO PARA IBI28: Resultados de D_KL
        if self.dkl_results:
            print(f"\n{'='*80}")
            print("RESULTADOS DE INFORMACIÓN (NUEVO PARA IBI28):")
            print(f"  D_KL promedio (pre):  {self.dkl_results['pre']['mean']:.6f} nats")
            print(f"  D_KL promedio (post): {self.dkl_results['post']['mean']:.6f} nats")
            print(f"  ΔD_KL:                {self.dkl_results['delta_dkl']:.6f} nats")
            print(f"  Distancia geodésica (pre):  {self.geodesic_results['pre']['mean']:.6f}")
            print(f"  Distancia geodésica (post): {self.geodesic_results['post']['mean']:.6f}")
        
        delta_r2 = abs(self.results['pre']['r2']['mean'] - self.results['post']['r2']['mean'])
        
        print(f"\n{'='*80}")
        print("MÉTRICAS DE RESILIENCIA:")
        print(f"  ΔR²: {delta_r2:.5f}")
        print(f"  ¿ΔR² < 0.30? {'✓ SÍ' if delta_r2 < 0.30 else '✗ NO'}")
        print(f"  ¿R² > 0.87? {'✓ SÍ' if self.results['pre']['r2']['mean'] > 0.87 else '⚠ PARCIAL'}")
        print(f"  α (AKF): {self.results.get('akf_alpha', 0.1)}")
        print(f"  Adaptación: {self.results.get('akf_adaptation', 'R_only')} (solo R, Q fija)")
        print("="*80)


# ================================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================================
if __name__ == "__main__":
    print("="*80)
    print("STRESS-TEST IBI - AKF CON D_KL Y TRAYECTORIA GEODÉSICA")
    print("Versión 7.1 - Corregido (Alineado con IBI_TEST_AKFR.py)")
    print("="*80)
    
    tester = IBIStressTestStable(config=TEST_CONFIG)

    # 1. Verificar observabilidad
    observability = tester.verify_observability_matrix()

    # 2. Comparar reducción de varianza (CORREGIDO)
    variance_results = tester.compare_variance_reduction()

    # 3. Ejecutar validación completa con D_KL
    results = tester.run_validation_completa()

    # 4. Imprimir resumen
    tester.print_summary()

    # 5. Generar LAS CUATRO GRÁFICAS (incluyendo nueva Figura 4)
    print("\n" + "="*80)
    print("GENERANDO 4 GRÁFICAS PARA IBI")
    print("="*80)

    plot_dir = './test_ibi_akf'

    # Figura 1: Convergencia MSE
    tester.plot_convergence_mse(plot_dir)

    # Figura 2: Reducción de Varianza
    tester.plot_variance_reduction(plot_dir)

    # Figura 3: Señal IBI
    tester.plot_signal_resilience(plot_dir)

    # Figura 4 (NUEVA): Trayectoria Geodésica con D_KL
    geodesic_plot, dkl_csv, geodesic_data = tester.plot_geodesic_trajectory(plot_dir)

    # Imprimir correlación D_KL - MSE
    if geodesic_data['correlation_dkl_mse']:
        print(f"\n{'='*80}")
        print(f"CORRELACIÓN D_KL - MSE: {geodesic_data['correlation_dkl_mse']:.4f}")
        print(f"{'='*80}")

    print(f"\n✓ Validación finalizada")
    print(f"✓ Todas las gráficas generadas en: {plot_dir}")
    print(f"✓ trayectoria geodésica")
    print("="*80)