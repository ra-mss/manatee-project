# src/predict.py

import torch
import librosa
import numpy as np
import argparse

from model import get_model

def preprocess_single_audio(file_path, sample_rate=22050, duration_s=0.5):
    """
    Preprocesa un único archivo de audio para la predicción
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        
        target_samples = int(duration_s * sr)
        # Si el audio es más largo, toma el centro; si es más corto, lo rellena.
        if len(y) > target_samples:
            start = (len(y) - target_samples) // 2
            y = y[start : start + target_samples]
        
        y = librosa.util.fix_length(y, size=target_samples)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)

        tensor = torch.from_numpy(log_S).float()
        tensor = tensor.unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)
        tensor = tensor.unsqueeze(0)

        return tensor
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar la arquitectura del modelo
    model = get_model(pretrained=False).to(device)
    # Cargar los pesos entrenados
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # Poner el modelo en modo de evaluación
    model.eval()

    print(f"Analizando archivo: {args.audio_file}")
    input_tensor = preprocess_single_audio(args.audio_file)

    if input_tensor is not None:
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
        
        probability = torch.sigmoid(output).item()
        prediction = "Manatí" if probability > args.threshold else "Ruido"

        print("\nResultado ")
        print(f"Predicción: {prediction}")
        print(f"Confianza (de ser manatí): {probability * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clasificar un nuevo archivo de audio.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--audio_file', type=str, required=True, help='Ruta al archivo de audio (.wav) a clasificar')
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral de probabilidad para clasificar como "Manatí"')
    
    args = parser.parse_args()
    main(args)