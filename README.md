# 🚗 Detección y Seguimiento de Vehículos con YOLO y Flujo Óptico

Sistema completo de visión por computador que integra técnicas avanzadas de detección y seguimiento de objetos para resolver un problema práctico: **el conteo automático de vehículos en videos de tráfico**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Descripción

Este proyecto implementa un sistema completo que combina:

- **Detección de objetos** mediante YOLO v8 (modelo pre-entrenado en COCO)
- **Seguimiento de objetos** mediante flujo óptico Lucas-Kanade
- **Aplicación práctica**: Conteo automático de vehículos que cruzan una línea virtual

El sistema procesa videos de tráfico en tiempo real, detecta vehículos (automóviles, motocicletas, autobuses, camiones), los sigue entre múltiples fotogramas manteniendo su identidad, y cuenta automáticamente aquellos que cruzan una línea virtual definida.

---

## ✨ Características

- ✅ **Detección en tiempo real** con YOLO v8
- ✅ **Seguimiento robusto** con flujo óptico Lucas-Kanade
- ✅ **Conteo preciso** de vehículos mediante línea virtual
- ✅ **Visualización completa** con bounding boxes, trayectorias e IDs
- ✅ **Evaluación cuantitativa** con métricas detalladas
- ✅ **Pipeline integrado** de extremo a extremo
- ✅ **Código documentado** y fácil de entender

---

## 🎯 Objetivos del Trabajo

1. Implementar y configurar un modelo YOLO (v8) para detectar vehículos
2. Aplicar técnicas de flujo óptico (Lucas-Kanade) para seguimiento
3. Integrar ambas técnicas en un pipeline coherente
4. Resolver aplicación práctica: conteo de vehículos
5. Evaluar cuantitativamente el desempeño del sistema

---

## 🛠️ Requisitos

### Software

- **Python** 3.10 o superior
- **pip** (gestor de paquetes de Python)

### Hardware Recomendado

- **CPU**: Procesador moderno (Intel i5 o equivalente)
- **RAM**: Mínimo 8 GB (recomendado 16 GB)
- **GPU**: Opcional pero recomendada para procesamiento más rápido (NVIDIA con CUDA)

---

## 📦 Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd trabajo4
```

### 2. Crear entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import cv2, numpy, ultralytics; print('✅ Todas las dependencias instaladas correctamente')"
```

---

## 🚀 Uso

### Opción 1: Usar el Notebook Jupyter (Recomendado)

1. **Abrir el notebook**:
   ```bash
   jupyter notebook notebooks/1_yolo_objetos.ipynb
   ```

2. **Ejecutar las celdas en orden**:
   - Las celdas están organizadas secuencialmente
   - Cada sección incluye explicaciones y visualizaciones
   - Los resultados se guardan automáticamente en `results/`

3. **Ver resultados**:
   - Video procesado: `results/video_procesado.mp4`
   - Frames individuales: `results/frame_*.jpg`
   - Estadísticas: `results/estadisticas.json`
   - Gráficas: `results/*.png`

### Opción 2: Usar el script Python

```bash
python src/yolo_objetos.py
```

### Configuración Básica

El sistema está configurado por defecto para:
- **Modelo YOLO**: `yolov8n.pt` (nano - más rápido)
- **Umbral de confianza**: 0.25
- **Umbral IoU**: 0.45
- **Video**: `data/SampleVideo_LowQuality.mp4`

### Personalización

Puedes modificar los parámetros en el notebook:

```python
# Cambiar modelo YOLO (más preciso pero más lento)
tracker = VehicleTracker(yolo_model='yolov8s.pt', conf_threshold=0.3)

# Procesar video completo (sin límite de frames)
stats = tracker.process_video(
    video_path=video_path,
    output_path=output_video,
    max_frames=None,  # Procesar todo el video
    save_frames=True
)

# Definir línea de conteo personalizada
custom_line = (100, 200, 800, 200)  # (x1, y1, x2, y2)
tracker.set_count_line(custom_line)
```

---

## 📊 Resultados

### Estadísticas del Procesamiento

Basado en el procesamiento de 300 frames del video de prueba:

| Métrica | Valor |
|---------|-------|
| **Frames procesados** | 300 |
| **Detecciones totales** | 4,274 |
| **Promedio detecciones/frame** | 14.25 |
| **Objetos únicos seguidos** | 114 |
| **Vehículos contados** | 45 |
| **FPS de procesamiento** | 4.33 |
| **Tiempo promedio/frame** | 230.77 ms |

### Visualizaciones Generadas

El sistema genera automáticamente múltiples visualizaciones:

- **`ejemplo_deteccion_yolo.png`**: Comparación frame original vs. con detecciones
- **`explicacion_iou.png`**: Visualización del concepto Intersection over Union
- **`explicacion_linea_virtual.png`**: Ejemplos de detección de cruce
- **`diagrama_pipeline.png`**: Flujo completo del sistema
- **`analisis_resultados.png`**: Análisis completo de métricas
- **`estadisticas_procesamiento.png`**: Gráficas de rendimiento
- **`ejemplos_frames_procesados.png`**: Muestra de frames procesados
- **`video_info.png`**: Información del video de entrada

---

## 📁 Estructura del Proyecto

```
trabajo4/
├── notebooks/
│   ├── 1_yolo_objetos.ipynb    # Notebook principal con implementación completa
│   └── yolov8n.pt              # Modelo YOLO (descargado automáticamente)
├── src/
│   └── yolo_objetos.py         # Código fuente (opcional)
├── data/
│   ├── SampleVideo_LowQuality.mp4
│   ├── Sample_Video_HighQuality.mp4
│   └── [datasets YOLO para entrenamiento]
├── results/                     # Resultados y visualizaciones
│   ├── video_procesado.mp4     # Video con anotaciones
│   ├── frame_*.jpg             # Frames individuales guardados
│   ├── estadisticas.json       # Métricas en formato JSON
│   └── *.png                   # Gráficas y visualizaciones
├── GITHUBPAGES/                # Informe y documentación
│   ├── informe.md              # Informe completo del proyecto
│   ├── index.md                # Página principal
│   ├── _config.yml             # Configuración Jekyll
│   └── results/                # Imágenes para el informe
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

---

## 🔧 Tecnologías Utilizadas

### Librerías Principales

- **[OpenCV](https://opencv.org/)** (cv2): Procesamiento de imágenes, video y algoritmos de visión por computador
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: Modelo de detección de objetos YOLO v8
- **[NumPy](https://numpy.org/)**: Operaciones numéricas y arrays multidimensionales
- **[Matplotlib](https://matplotlib.org/)**: Visualización de datos y gráficas
- **[PyTorch](https://pytorch.org/)**: Framework de deep learning (requerido por YOLO)

### Algoritmos Implementados

- **YOLO v8**: Detección de objetos en tiempo real
- **Lucas-Kanade Optical Flow**: Seguimiento de objetos entre frames
- **Intersection over Union (IoU)**: Asociación de detecciones
- **Geometría analítica**: Detección de cruce de línea virtual

---

## 🎓 Información Académica

**Curso:** Visión por Computador – 3009228  
**Semestre:** 2025-02  
**Universidad:** Universidad Nacional de Colombia  
**Facultad:** Facultad de Minas  
**Departamento:** Ciencias de la Computación y de la Decisión

**Trabajo:** Detección y Seguimiento de Objetos con YOLO y Flujo Óptico

---


## 🔍 Características Técnicas

### Detección

- **Modelo**: YOLO v8 nano (yolov8n.pt)
- **Clases detectadas**: Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Umbral de confianza**: 0.25
- **NMS IoU threshold**: 0.45

### Seguimiento

- **Método**: Flujo óptico Lucas-Kanade
- **Ventana de búsqueda**: 15×15 píxeles
- **Niveles de pirámide**: 2
- **Asociación IoU threshold**: 0.3
- **Frames sin ver (máx)**: 10

### Conteo

- **Línea virtual**: Configurable (por defecto: horizontal en el centro)
- **Algoritmo**: Detección de cambio de signo en ecuación de línea
- **Prevención doble conteo**: Flag `crossed_line` por objeto

---

## 🐛 Solución de Problemas

### Error: "ultralytics no está instalado"

```bash
pip install ultralytics
```

### Error: "No se pudo abrir el video"

- Verifica que el archivo de video existe en `data/`
- Asegúrate de que el formato es compatible (MP4 recomendado)
- Verifica permisos de lectura del archivo

### Procesamiento muy lento

- Considera usar un modelo YOLO más pequeño (`yolov8n.pt`)
- Reduce la resolución del video
- Usa GPU si está disponible
- Limita el número de frames procesados (`max_frames`)

### Detecciones faltantes

- Reduce el `conf_threshold` (ej: 0.15)
- Usa un modelo YOLO más grande (`yolov8s.pt` o `yolov8m.pt`)
- Verifica que los vehículos sean visibles y de tamaño adecuado

---

## 📝 Notas

- El modelo YOLO se descarga automáticamente la primera vez que se ejecuta
- Los resultados se guardan automáticamente en `results/`
- El sistema procesa videos en formato MP4, AVI, o formatos compatibles con OpenCV
- Para mejor rendimiento, se recomienda usar GPU

---

## 🤝 Contribuciones

Este es un proyecto académico. Si encuentras errores o tienes sugerencias:

1. Abre un issue describiendo el problema
2. Proporciona información sobre tu entorno (OS, Python version, etc.)
3. Incluye mensajes de error completos si aplica

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico. El código está disponible para fines educativos.

---

## 🙏 Agradecimientos

- **Ultralytics** por el modelo YOLO v8
- **OpenCV** por las herramientas de visión por computador
- **Universidad Nacional de Colombia** por el apoyo académico


**Desarrollado para el Trabajo 4: Detección y Seguimiento de Objetos con YOLO y Flujo Óptico**  
**Curso:** Visión por Computador – 3009228  
**Universidad Nacional de Colombia – Facultad de Minas (2025-02)**

