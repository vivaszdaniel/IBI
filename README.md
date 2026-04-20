# IBI: Instrumentación Basada en Información (Core)

![Status](https://img.shields.io/badge/Status-Research--Active-blue)
![Platform](https://img.shields.io/badge/Platform-ESP32--S3%20|%20Python-orange)
![Unit](https://img.shields.io/badge/Organization-Unidad%20Proyecto%20Aragua%20(UPA)-green)

**IBI (Information-Based Instrumentation)** es un marco metrológico estocástico desarrollado en la **Unidad Proyecto Aragua (UPA)** de la Universidad de Carabobo. Redefine la caracterización de materia mediante la integración de física de la información, geometría de la información y algoritmos de estimación de estado informados por la física.

## 🔬 Visión General
Este repositorio contiene el núcleo algorítmico diseñado para extraer información coherente en sistemas iónicos complejos. A diferencia de la medición pasiva, IBI utiliza un enfoque activo basado en el formalismo de proyección de **Mori-Zwanzig** y el **Filtro de Kalman con Estados Aumentados (AKF)**.

### Pilares Tecnológicos:
- **Física de la Información:** Aplicación del Hamiltoniano de interacción para separar la señal de la memoria del sistema.
- **Geometría de la Información:** Uso de la **Matriz de Fisher** para maximizar la distinguibilidad entre analitos.
- **Eficiencia Metrológica ($\eta_{IBI}$):** Una métrica universal para cuantificar la calidad de la extracción de información.
- **Optimización para Edge Computing:** Algoritmos diseñados específicamente para el procesador Xtensa® del ESP32-S3.

## 📂 Estructura del Repositorio
- `/python`: Suite completa de validación numérica (AKF, UKF, EKF, LKF).
  - `simulacion_completa_ibi.py`: Generador de métricas teóricas y tablas LaTeX.
  - `test_de_validacion_*.py`: Pruebas de estrés y trayectoria geodésica.
- `/docs`: Documentación técnica y preprints de la Unidad Proyecto Aragua.
- `/firmware`: (En desarrollo) Implementación en C++/TinyML para microcontroladores.

## 🚀 Validación Numérica
El sistema ha sido validado comparando diferentes arquitecturas de filtrado. Los resultados demuestran que el **AKF (Augmented Kalman Filter)** de IBI ofrece una reducción de varianza superior en sistemas con memoria no markoviana.

Para ejecutar las validaciones:
```bash
pip install -r requirements.txt
python python/simulacion_completa_ibi.py
python python/test_de_validacion_akf.py
python python/test_de_validacion_ekf.py
python python/test_de_validacion_lkf.py
python python/test_de_validacion_ukf.py
