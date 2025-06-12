import os
from PIL import Image
import imagehash

# === CONFIGURAÇÕES GLOBAIS === #
INPUT_DIRS = ["./C1", "./C2", "./C3"]
OUTPUT_FILE = "./duplicates.txt"  # Caminho para o arquivo de saída

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
    return parts[0]

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
                else:
                    hash_dict[img_hash] = file_path

    return duplicates

# Função principal
def main():
    duplicates = find_duplicates()

    # Salvar as duplicatas em um arquivo de texto
    with open(OUTPUT_FILE, 'w') as f:
        if duplicates:
            f.write("Imagens duplicadas encontradas entre as pastas C1, C2 e C3:\n")
            for dup in duplicates:
                f.write(f"[DUPLICADO] {dup[0]} é duplicado de {dup[1]}\n")
        else:
            f.write("Nenhuma imagem duplicada encontrada entre as pastas C1, C2 e C3.\n")

    print(f"Resultado salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
