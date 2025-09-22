# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
import argparse

from model import get_model
from preprocess import load_and_split_data, process_filepaths

def main(args):
    # Carga y División de Datos
    print("Cargando y dividiendo los datos")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        load_and_split_data(args.data_path)

    # Preprocesamiento: Creación de Espectrogramas
    print("Creando espectrogramas para el conjunto de entrenamiento")
    X_train, y_train = process_filepaths(train_files, train_labels)
    
    print("Creando espectrogramas para el conjunto de validación")
    X_val, y_val = process_filepaths(val_files, val_labels)

    # Preparación para PyTorch
    print("Preparando DataLoaders de PyTorch")
    # Añadir un canal y convertir a tensores de PyTorch
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_val_tensor = torch.from_numpy(X_val).unsqueeze(1).float()
    y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)

    # Repetir el canal para que sean 3 canales
    X_train_tensor = X_train_tensor.repeat(1, 3, 1, 1)
    X_val_tensor = X_val_tensor.repeat(1, 3, 1, 1)

    # Crear datasets y dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Configuración del Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    model = get_model(pretrained=True).to(device)
    
    # Función de pérdida y optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': args.lr_finetune},
        {'params': model.fc.parameters(), 'lr': args.lr_head}
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Pipeline de aumento de datos
    augmentation_pipeline = None
    if args.use_augmentation:
        print("Usando aumento de datos (augmentation)")
        augmentation_pipeline = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=8),
            T.TimeMasking(time_mask_param=16)
        ).to(device)

    # Bucle de Entrenamiento
    best_val_accuracy = 0.0
    print("Iniciando entrenamiento")
    for epoch in range(args.epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if args.use_augmentation:
                inputs = augmentation_pipeline(inputs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                predicted = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        validation_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        
        # Guardar el mejor modelo
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), args.model_output_path)
            print(f"--> Mejor modelo guardado con {accuracy:.2f}% de precisión en '{args.model_output_path}'")
        
        # Actualizar el planificador
        scheduler.step(validation_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación de sonidos de manatí')
    parser.add_argument('--data_path', type=str, required=True, help='Ruta a la carpeta con los datos (debe contener subcarpetas MV y Noise)')
    parser.add_argument('--model_output_path', type=str, default='best_model.pth', help='Ruta donde se guardará el mejor modelo')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas para entrenar')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del lote')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Tasa de aprendizaje para la capa final')
    parser.add_argument('--lr_finetune', type=float, default=1e-5, help='Tasa de aprendizaje para las capas descongeladas')
    parser.add_argument('--use_augmentation', action='store_true', help='Activar el aumento de datos')
    
    args = parser.parse_args()
    main(args)