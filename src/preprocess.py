# src/preprocess.py

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_split_data(data_path, test_size=0.2, val_size=0.1):
    """
    Carga las rutas de los archivos y las divide en conjuntos de 
    entrenamiento, validación y prueba
    """
    # Rutas a las carpetas
    path_mv = os.path.join(data_path, 'MV')
    path_noise = os.path.join(data_path, 'Noise')
    
    # Crear una lista de todas las rutas de archivo y sus etiquetas
    filepaths = [os.path.join(path_mv, f) for f in os.listdir(path_mv)] + \
                [os.path.join(path_noise, f) for f in os.listdir(path_noise)]
    
    # Etiqueta 1 para Manatí (MV), Etiqueta 0 para Ruido (Noise)
    labels = [1] * len(os.listdir(path_mv)) + [0] * len(os.listdir(path_noise))
    
    # División 80/20 para entrenamiento y el resto
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # El 20% restante en 10% para validación y 10% para prueba
    relative_val_size = val_size / (1 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=relative_val_size, random_state=42, stratify=train_val_labels
    )
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

def create_mel_spectrogram(filepath, sample_rate=22050):
    """Carga un archivo de audio y lo convierte en un mel-espectrograma"""
    try:
        y, sr = librosa.load(filepath, sr=sample_rate)
        # Todos los clips con misma longitud
        y = librosa.util.fix_length(y, size=sr // 2) # // 2 porque los clips son de 0.5s
        
        # Generar mel-espectrograma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        
        # Convertir a decibelios (dB) para una mejor representación
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return None

def process_filepaths(filepaths, labels):
    """
    Convierte una lista de rutas de archivo en un array de espectrogramas y etiquetas
    """
    spectrograms = np.array([create_mel_spectrogram(fp) for fp in tqdm(filepaths)])
    return spectrograms, np.array(labels)