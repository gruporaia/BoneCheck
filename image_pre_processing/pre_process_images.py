import os
import cv2
import csv
import shutil
import numpy as np
import re
import unicodedata
from PIL import Image, ImageEnhance
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from glob import glob
import imagehash

# === CONFIGURAÇÕES GLOBAIS === #
INPUT_DIRS = ["./C1", "./C2", "./C3"]
OUTPUT_DIR = "./processed"
MULTICLASS_IMAGES_DIR = "./multiclass_images"  # Novo diretório para imagens duplicadas
CSV_FILE = os.path.join(OUTPUT_DIR, "image_metadata.csv")
CROP_PARAMS = {'left': 0.04, 'right': 0.04, 'top': 0.10, 'bottom': 0.04}
BRIGHTNESS_FACTOR = 1.5
ZOOM_FACTOR = 1.2
LABEL_MAP = {"C1": 0, "C2": 1, "C3": 2}
OUTPUT_DUPLICATES_FILE = "./duplicates.txt"  # Caminho para o arquivo de duplicatas

# === FUNÇÕES DE PROCESSAMENTO DE IMAGENS === #

def normalize_name(fname):
    name = os.path.splitext(fname)[0]
    name = re.sub(r'^\d+_?', '', name)  # Remove prefixo numérico
    name = re.sub(r'px from|panorama|ok|s\d|i\d|\(|\)|-|_', '', name, flags=re.IGNORECASE)  # Remove palavras irrelevantes
    name = ''.join(c for c in unicodedata.normalize('NFKD', name) if unicodedata.category(c) != 'Mn')  # Remove acentos
    name = re.sub(r'\s+', '', name).strip().lower()  # Remove espaços extras e deixa minúsculo
    return name

def is_duplicate(fname, name_tracker, file_path):
    person_key = normalize_name(fname)
    if person_key in name_tracker:
        print(f"[DUPLICADO] {file_path}")
        return True
    name_tracker.add(person_key)
    return False

def crop_borders(img):
    h, w = img.shape[:2]
    top = int(h * CROP_PARAMS['top'])
    bottom = h - int(h * CROP_PARAMS['bottom'])
    left = int(w * CROP_PARAMS['left'])
    right = w - int(w * CROP_PARAMS['right'])
    return img[top:bottom, left:right]

def get_rois(img):
    h, w = img.shape[:2]
    h_half, w_half = h // 2, w // 2
    return {
        'top_left': img[:h_half, :w_half],
        'top_right': img[:h_half, w_half:],
        'bottom_left': img[h_half:, :w_half],
        'bottom_right': img[h_half:, w_half:]
    }

def augment_image(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bright = ImageEnhance.Brightness(pil_img).enhance(BRIGHTNESS_FACTOR)
    zoomed = pil_img.resize((int(pil_img.width * ZOOM_FACTOR), int(pil_img.height * ZOOM_FACTOR)))
    return {
        '': img,
        '_bright': cv2.cvtColor(np.array(bright), cv2.COLOR_RGB2BGR),
        '_zoom': cv2.cvtColor(np.array(zoomed), cv2.COLOR_RGB2BGR),
    }

def save_image(img, name):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Salvando no formato RGB

def process_image(file_path, fname, img_id, class_label):
    metadata = []
    try:
        img_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Checagem robusta contra imagens inválidas
        if img is None or img.shape[0] == 0 or img.shape[1] == 0 or not hasattr(img, 'shape'):
            print(f"[CORROMPIDO] {file_path.name}")
            return []

        # Checagem extra para JPEGs com erros silenciosos
        try:
            Image.open(file_path).verify()  # PIL tenta validar
        except Exception:
            print(f"[JPEG CORROMPIDO] {file_path.name}")
            return []

        id_str = str(img_id).zfill(5)

        # 1. Original
        save_image(img, f"{id_str}.png")

        # 2. Crop borda
        cropped = crop_borders(img)
        save_image(cropped, f"{id_str}_cropped.png")

        # 3. ROIs
        rois = get_rois(cropped)
        for roi_key, roi_img in rois.items():
            roi_name = f"{id_str}_cropped_{roi_key}.png"
            save_image(roi_img, roi_name)

            # 4. Augmentações para cada ROI
            augmented = augment_image(roi_img)
            for suffix, aug_img in augmented.items():
                aug_name = f"{id_str}_cropped_{roi_key}{suffix}.png"
                save_image(aug_img, aug_name)

        relative_path = os.path.relpath(file_path, start=os.getcwd())
        numeric_label = LABEL_MAP[class_label]
        metadata.append((fname, id_str, class_label, numeric_label, relative_path, img.shape[1], img.shape[0]))
        return metadata

    except Exception as e:
        print(f"[ERRO] {file_path}: {e}")
        return []

def generate_csv(metadata):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Nome Original', 'ID', 'Classe', 'Classe Numérica', 'Caminho Original', 'Largura', 'Altura'])
        writer.writerows(metadata)

# === DETECÇÃO DE DUPLICATAS === #

# Função para calcular o hash perceptual de uma imagem
def get_image_hash(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # Garantir que a imagem esteja em RGB
        hash_value = imagehash.phash(img)  # Usando perceptual hashing (phash)
        return hash_value
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None

# Função para obter a pasta raiz (C1, C2, ou C3) da imagem, ignorando subpastas
def get_root_folder(file_path):
    parts = file_path.split(os.sep)
    return parts[0]  # Extrai a primeira parte do caminho, que deve ser C1, C2 ou C3

# Função para verificar duplicatas
def find_duplicates():
    hash_dict = {}  # Dicionário para armazenar o hash e o caminho da imagem
    duplicates = []  # Lista de imagens duplicadas

    # Caminhar por todas as pastas e subpastas
    for class_path in INPUT_DIRS:
        for root, _, files in os.walk(class_path):
            for fname in files:
                # Verificar se é uma imagem (formatos suportados)
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')):
                    continue

                file_path = os.path.join(root, fname)
                
                # Obter o hash da imagem
                img_hash = get_image_hash(file_path)
                if img_hash is None:
                    continue

                # Obter a pasta raiz de onde a imagem vem (C1, C2 ou C3)
                root_folder = get_root_folder(file_path)  # C1, C2, ou C3

                # Se o hash já estiver no dicionário e for de uma pasta diferente, é uma duplicata
                if img_hash in hash_dict:
                    original_path = hash_dict[img_hash]
                    original_root_folder = get_root_folder(original_path)

                    # Verificar se a imagem está em pastas diferentes (C1, C2, C3)
                    if original_root_folder != root_folder:
                        duplicates.append((file_path, original_path))
                        # Mover a imagem duplicada para a pasta "multiclass_images"
                        duplicate_dst = os.path.join(MULTICLASS_IMAGES_DIR, os.path.basename(file_path))
                        os.makedirs(MULTICLASS_IMAGES_DIR, exist_ok=True)
                        shutil.move(file_path, duplicate_dst)  # Mover a imagem para o diretório de duplicatas
                else:
                    hash_dict[img_hash] = file_path

    return duplicates

# Função para salvar as duplicatas em um arquivo TXT
def save_duplicates(duplicates):
    with open(OUTPUT_DUPLICATES_FILE, 'w') as f:
        if duplicates:
            f.write("Imagens duplicadas encontradas entre as pastas C1, C2 e C3:\n")
            for dup in duplicates:
                f.write(f"[DUPLICADO] {dup[0]} é duplicado de {dup[1]}\n")
        else:
            f.write("Nenhuma imagem duplicada encontrada entre as pastas C1, C2 e C3.\n")
    print(f"Resultado de duplicatas salvo em: {OUTPUT_DUPLICATES_FILE}")

# Função para realizar o K-Fold
def perform_kfold_split(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['Fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Classe Numérica'])):
        df.loc[val_idx, 'Fold'] = fold
    return df

# Função para mover os arquivos para as pastas de cada fold
def move_files_to_folds(df, base_dir):
    for _, row in df.iterrows():
        fold_dir = os.path.join(base_dir, f"fold_{row['Fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Encontrar todos os arquivos que começam com o ID (ex: 01148*.png)
        pattern = os.path.join(OUTPUT_DIR, f"{row['ID']}*.png")
        matching_files = glob(pattern)
        
        if not matching_files:
            print(f"Nenhum arquivo encontrado para o padrão: {pattern}")
        
        for file_path in matching_files:
            shutil.move(file_path, fold_dir)

# === FUNÇÃO PRINCIPAL === #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MULTICLASS_IMAGES_DIR, exist_ok=True)  # Criando a pasta para duplicatas
    image_id = 1
    all_metadata = []
    name_tracker = set()

    # Encontra duplicatas antes de começar o processamento
    duplicates = find_duplicates()
    save_duplicates(duplicates)

    # Processamento de Imagens
    for class_path in INPUT_DIRS:
        class_label = os.path.basename(class_path)

        for root, _, files in os.walk(class_path):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')):
                    continue

                fpath = os.path.join(root, fname, )

                # Processa a imagem e ignora duplicatas
                metadata = process_image(fpath, fname, image_id, class_label)
                if metadata:
                    all_metadata.extend(metadata)
                    image_id += 1

    generate_csv(all_metadata)
    print(f"Processamento de imagens finalizado. {image_id - 1} imagens processadas.")

    # Carregar o CSV gerado
    df = pd.read_csv(CSV_FILE)

    # Remover imagens duplicadas da lista de metadata
    df = df[~df['Caminho Original'].isin([dup[0] for dup in duplicates])]

    # Realizar o K-Fold
    df = perform_kfold_split(df)
    df.to_csv('processed/image_metadata_with_folds.csv', index=False)

    print('K-fold split concluído. Salvado em image_metadata_with_folds.csv.')

    # Mover os arquivos para suas respectivas dobras
    processed_dir = './processed'
    df['ID'] = df['ID'].apply(lambda x: str(x).zfill(5))  # Garantir que o ID esteja com 5 dígitos
    move_files_to_folds(df, processed_dir)

    # Organizar o CSV final
    df = df.drop(columns=["Nome Original", "Caminho Original"])
    df["ID"] = df["ID"].apply(lambda x: str(x).zfill(5) + ".png")
    df = df.rename(columns={"ID": "path", "Classe": "class", "Classe Numérica": "class_numeric", "Largura": "width", "Altura": "height", "Fold": "fold"})
    df.to_csv("processed/data_with_folds.csv", index=False)

    print('Arquivos movidos para as respectivas pastas de dobragem.')

if __name__ == "__main__":
    main()

