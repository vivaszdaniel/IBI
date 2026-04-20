"""
TEST DE VALIDACIÓN IBI - AKF (ADAPTIVE KALMAN FILTER)
Autor: Daniel Vivas 
Email: dvivas1@uc.edu.ve
Afiliacion: Universidad de Carabobo, Venezuela
Fecha: Marzo 2026
Genera:
    convergencia_mse_TIMESTAMP.png - MSE vs. Tiempo (Figura 1 del paper)
    reduccion_varianza_TIMESTAMP.png - Ratio vs. Muestras (Figura 2 del paper)
    senal_ibi_TIMESTAMP.png - Resiliencia IBI (Figura 3 del paper)
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
import os

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
}

PHYSICAL_CONSTANTS = {
    'F': 96485.332,
    'R': 8.314,
    'T': 298.15,
    'kB': 1.381e-23,
}

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
    
    def filter(self, measurements):
        """Aplicar filtro de Kalman Adaptativo completo"""
        n_meas = len(measurements)
        x_est = np.zeros((n_meas, self.n_states))
        P = np.eye(self.n_states) * 0.1
        
        # Inicialización
        x_est[0, 0] = np.mean(measurements[:min(5, len(measurements))])
        x_est[0, 1] = 0.0
        x_est[0, 2] = 0.0
        
        # Reiniciar R adaptativo para cada nueva filtración
        self.R_anterior = self.R_base
        
        for k in range(1, n_meas):
            # ========== PREDICCIÓN ==========
            x_pred = self.F @ x_est[k-1]
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # ========== INNOVACIÓN (RESIDUO) ==========
            z_k = measurements[k]
            residuo = z_k - (self.H @ x_pred)  # Innovación del sensor
            
            # ========== ADAPTACIÓN DE R (LÓGICA AKF) ==========
            # Si el residuo es grande, aumenta R para "dudar" del sensor ruidoso
            R_adaptativo = (1 - self.alpha) * self.R_anterior + self.alpha * (residuo**2)
            
            # Limitar R_adaptativo para evitar inestabilidad numérica
            R_adaptativo = np.clip(R_adaptativo, self.R_base * 0.1, self.R_base * 1000)
            
            # Actualizar R_anterior para siguiente iteración
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
        
        return x_est

# ================================================================================
# CLASE PRINCIPAL DE VALIDACIÓN (ACTUALIZADA PARA AKF)
# ================================================================================
class IBIStressTestStable:
    """Test de validación con Filtro Kalman Adaptativo (AKF)"""
    def __init__(self, config=TEST_CONFIG):
        self.config = config
        self.n_samples = config['n_samples']
        self.dt = config['dt']
        self.time_interference = self.n_samples // 2
        self.t = np.linspace(0, self.n_samples * self.dt, self.n_samples)
        
        # AKF con parámetros de adaptación
        self.kalman_filter = AdaptivePhysicsInformedKalmanFilter(
            dt=self.dt, 
            alpha=0.1,  # Factor de suavizado AKF
            R_base=1e-5
        )
        self.results = None
        self.observability_results = None
        self.variance_comparison = None

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

    def kalman_inference_engine(self, raw_signal):
        """Motor Kalman Adaptativo"""
        x_est = self.kalman_filter.filter(raw_signal)
        return x_est[:, 0]

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
        """Comparar reducción de varianza"""
        print("\n" + "="*80)
        print("COMPARACIÓN DE REDUCCIÓN DE VARIANZA")
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
                y_kalman = self.kalman_inference_engine(y_noisy)
                
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
        """Ejecutar validación completa"""
        if n_iterations is None:
            n_iterations = self.config['n_iterations']
        
        np.random.seed(RANDOM_SEED)
        
        metrics_pre = {'r2': [], 'rmse': [], 'mse': [], 'mae': [], 'mape': []}
        metrics_post = {'r2': [], 'rmse': [], 'mse': [], 'mae': [], 'mape': []}
        
        print(f"\nEjecutando {n_iterations} iteraciones de Monte Carlo...")
        
        for i in range(n_iterations):
            y_clean = self.generate_clean_signal()
            y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
            y_pred = self.kalman_inference_engine(y_noisy)
            
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
            
            if not np.isnan(m_post['r2']) and np.isfinite(m_post['r2']):
                for key in metrics_post:
                    metrics_post[key].append(m_post[key])
        
        def summarize(metrics_dict):
            result = {}
            for key, values in metrics_dict.items():
                if len(values) > 0:
                    # CORRECCIÓN CRÍTICA: IC95 = 1.96 * std / √n (intervalo de confianza de la media)
                    # No: 1.96 * std (que es banda de dispersión, no IC de la media)
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
            'akf_alpha': 0.1,  # Registrar parámetro AKF
            'akf_adaptation': 'R_only'  # Especificar que solo R es adaptativo
        }
        
        return self.results

    def export_results(self, output_dir='./test_ibi_akf'):
        """Exportar resultados"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_path = os.path.join(output_dir, f'resultados_ibi_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'observability': self.observability_results,
                'variance_comparison': self.variance_comparison,
                'physical_constants': PHYSICAL_CONSTANTS
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON exportado: {json_path}")
        
        csv_path = os.path.join(output_dir, f'metricas_ibi_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Fase', 'Métrica', 'Media', 'Std', 'IC95', 'Min', 'Max'])
            
            for fase in ['pre', 'post']:
                for metrica in ['r2', 'rmse', 'mae', 'mape']:
                    row = [
                        fase, metrica,
                        self.results[fase][metrica]['mean'],
                        self.results[fase][metrica]['std'],
                        self.results[fase][metrica]['ic95'],
                        self.results[fase][metrica]['min'],
                        self.results[fase][metrica]['max']
                    ]
                    writer.writerow(row)
        print(f"✓ CSV exportado: {csv_path}")
        
        return json_path, csv_path

    def plot_convergence_mse(self, output_dir='./test_ibi_akf'):
        """
        FIGURA 1: MSE vs. Tiempo
        CORRECCIÓN: Ahora compara Promediado Estándar vs. AKF (no sensor crudo vs. filtro)
        Esto alinea con el título de la Sec 5.1 del manuscrito.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generar datos de una corrida representativa para ver la "señal" del error
        y_clean = self.generate_clean_signal()
        y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
        y_kalman = self.kalman_inference_engine(y_noisy)
        y_avg = self.standard_averaging_estimator(y_noisy)
        
        # CORRECCIÓN: Calcular error del promediado estándar (no ruido crudo)
        err_avg = (y_avg - y_clean)**2
        err_kalman = (y_kalman - y_clean)**2
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.t * 1000, err_avg, color='orange', alpha=0.4, label='Error Promediado Estándar')
        plt.plot(self.t * 1000, err_kalman, color='red', linewidth=2, label='Error Filtro AKF (Propuesto)')
        
        plt.yscale('log')  # Escala logarítmica para ver la convergencia real
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

    def plot_variance_reduction(self, output_dir='./test_ibi_akf'):
        """
        FIGURA 2: Ratio de Mejora (Barras de desempeño)
        """
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
        plt.title('Figura 2: Ratio de Reducción de Varianza (AKF)', fontsize=13)
        
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
        """
        FIGURA 3: Comparación de las dos gráficas (Normal vs Filtrada)
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        y_clean = self.generate_clean_signal()
        y_noisy = self.generate_observed_signal(y_clean, snr_db=self.config['snr_db'])
        y_kalman = self.kalman_inference_engine(y_noisy)
        
        plt.figure(figsize=(14, 7))
        
        # Sombreado de zonas
        plt.axvspan(0, self.t[self.time_interference], color='gray', alpha=0.1, label='Operación Normal')
        plt.axvspan(self.t[self.time_interference], self.t[-1], color='red', alpha=0.05, label='Zona de Interferencia')
        
        # Señales
        plt.plot(self.t, y_noisy, color='orange', alpha=0.4, label='Señal Sensor (Ruidosa/Normal)', linewidth=1)
        plt.plot(self.t, y_clean, color='black', linestyle='--', alpha=0.6, label='Referencia Ideal')
        plt.plot(self.t, y_kalman, color='blue', label='Señal Filtrada (IBI-AKF)', linewidth=2.5)
        
        plt.title("Figura 3: Resiliencia y Filtrado de Señal IBI (AKF)", fontsize=15, fontweight='bold')
        plt.xlabel("Tiempo (segundos)", fontsize=12)
        plt.ylabel("Amplitud / Información (Ψ)", fontsize=12)
        
        # Ajuste dinámico de escala para que se vea la comparación
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
        """Imprimir resumen"""
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
    print("STRESS-TEST IBI - AKF (3 GRÁFICAS PARA PAPER)")
    print("Versión 6.1 - Correcciones Post-Revisión")
    print("="*80)
    
    tester = IBIStressTestStable(config=TEST_CONFIG)
    
    # 1. Verificar observabilidad
    observability = tester.verify_observability_matrix()

    # 2. Comparar reducción de varianza
    variance_results = tester.compare_variance_reduction()

    # 3. Ejecutar validación completa
    results = tester.run_validation_completa()

    # 4. Imprimir resumen
    tester.print_summary()

    # 5. Verificar criterios
    print("\n" + "="*80)
    print("VERIFICACIÓN DE CRITERIOS")
    print("="*80)

    delta_r2 = abs(results['pre']['r2']['mean'] - results['post']['r2']['mean'])

    criterios = {
        'ΔR² < 0.30': delta_r2 < 0.30,
        'R² > 0.87 (pre)': results['pre']['r2']['mean'] > 0.87,
        'N ≥ 95 corridas': results['n_valid_pre'] >= 95,
        'Sistema observable': observability['observable'],
    }

    all_passed = True
    for criterio, passed in criterios.items():
        print(f"  {'✓' if passed else '✗'} {criterio}")
        all_passed = all_passed and passed

    print(f"\n  VEREDICTO: {'✓ CUMPLE' if all_passed else '⚠ PENDIENTE'}")
    print("="*80)

    # 6. Exportar resultados
    if TEST_CONFIG['export_csv'] or TEST_CONFIG['export_json']:
        tester.export_results()

    # 7. Generar LAS TRES GRÁFICAS
    print("\n" + "="*80)
    print("GENERANDO 3 GRÁFICAS PARA PAPER")
    print("="*80)

    plot_dir = './test_ibi_akf'

    # Figura 1: Convergencia MSE (LA MÁS IMPORTANTE - CORREGIDA)
    tester.plot_convergence_mse(plot_dir)

    # Figura 2: Reducción de Varianza
    tester.plot_variance_reduction(plot_dir)

    # Figura 3: Señal IBI
    tester.plot_signal_resilience(plot_dir)

    print(f"\n✓ Validación finalizada")
    print(f"✓ Todas las gráficas generadas en: {plot_dir}")
    print("="*80)