# Clasificador de Vocalizaciones de Manatíes con Deep Learning

Este proyecto utiliza un modelo de Deep Learning (CNN con Transfer Learning) para clasificar clips de audio y determinar si contienen vocalizaciones de manatíes.

<img width="754" height="390" alt="espectrograma" src="https://github.com/user-attachments/assets/4a59391f-2a9f-4a0a-b2b9-58e6b12190f5" />

## Descripción del Proyecto

El monitoreo acústico es una herramienta no invasiva y efectiva para estudiar poblaciones de mamíferos marinos como los manatíes. Este proyecto aprovecha un dataset público de sonidos de manatíes para entrenar un modelo capaz de automatizar la tarea de identificación, sentando las bases para sistemas de monitoreo a gran escala.

## Estructura del Repositorio

```
.
├── data/                 # Carpeta para los datos de audio (no incluida en el repo)
├── models/               # Carpeta para guardar los modelos entrenados (.pth)
├── notebooks/            # Notebooks de Jupyter con la experimentación inicial
│   └── Manatee_Project.ipynb
├── src/                  # Código fuente modularizado
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este archivo
```

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/manatee-sound-classifier.git](https://github.com/tu-usuario/manatee-sound-classifier.git)
    cd manatee-sound-classifier
    ```

2.  **(Recomendado) Crea un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga los datos:**
    Descarga el dataset desde [EDI Data Portal](https://portal.edirepository.org/nis/mapbrowse?packageid=edi.2108.1) y descomprímelo en una carpeta `data/Manatee_fulltraining_final/`. La estructura debe quedar así:
    ```
    data/Manatee_fulltraining_final/
    ├── MV/
    │   └── ... (archivos .wav)
    └── Noise/
        └── ... (archivos .wav)
    ```

## Uso

### Entrenamiento
Para entrenar un nuevo modelo desde cero, ejecuta el script `train.py`.

-   **Para entrenar el Modelo A (sin aumento de datos):**
    ```bash
    python src/train.py --data_path ./data/Manatee_fulltraining_final/ --model_output_path ./models/model_A.pth --epochs 30
    ```
-   **Para entrenar el Modelo B (con aumento de datos):**
    ```bash
    python src/train.py --data_path ./data/Manatee_fulltraining_final/ --model_output_path ./models/model_B.pth --epochs 15 --use_augmentation
    ```

### Predicción
Para clasificar un nuevo archivo de audio (`.wav`) con un modelo ya entrenado:

```bash
python src/predict.py --model_path ./models/model_A.pth --audio_file /ruta/a/tu/audio.wav
```

## Resultados

Se entrenaron dos modelos principales con diferentes estrategias:

### Modelo A: "Especialista" (Sin aumento de datos)
Este modelo fue entrenado solo con los datos originales. Alcanza una alta precisión en datos "limpios" similares a los de entrenamiento, pero puede ser menos robusto en entornos ruidosos.

**Reporte de Clasificación (Conjunto de Prueba):**
```
              precision    recall  f1-score   support
       Noise       0.92      0.95      0.93      2345
     Manatee       0.93      0.89      0.91      1813
    accuracy                           0.92      4158
```

### Modelo B: "Generalista" (Con aumento de datos)
Este modelo fue entrenado con una versión "suave" de aumento de datos para mejorar su capacidad de generalización. Es extremadamente preciso cuando predice "Manatí" (98%), pero es más conservador y puede pasar por alto algunas vocalizaciones (menor recall).

**Reporte de Clasificación (Conjunto de Prueba):**
```
              precision    recall  f1-score   support
       Noise       0.69      0.99      0.81      2345
     Manatee       0.98      0.41      0.58      1813
    accuracy                           0.74      4158
```

## Agradecimientos 

Este proyecto fue posible gracias al siguiente dataset público:

**Título:**
Bioacoustic Dataset of African and Florida Manatee Vocalizations for Machine Learning Applications, 2020-2022

**Referencia:**
> Rycyk, A., V. Cargille, D. Bojali, C. Factheu, U. Ejimadu, C. Berchem, and A. Takoukam Kamla. 2025. Bioacoustic Dataset of African and Florida Manatee Vocalizations for Machine Learning Applications, 2020-2022 ver 1. Environmental Data Initiative. https://doi.org/10.6073/pasta/f6b14cae6916da998808a8395cf38ae0 (Accessed 2025-09-21).
