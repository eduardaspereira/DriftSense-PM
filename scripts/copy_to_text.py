#!/usr/bin/env python3
"""
Script para copiar todos os ficheiros de uma pasta para ficheiros .txt individuais.
Uso: python copy_to_txt.py <caminho_da_pasta>
"""

import argparse
import shutil
from pathlib import Path


def copy_files_to_txt(folder_path):
    """
    Copia todos os ficheiros de uma pasta para ficheiros .txt individuais em uma subpasta.
    
    Args:
        folder_path (str): Caminho da pasta a processar
    """
    folder = Path(folder_path)
    
    # Validar se a pasta existe
    if not folder.exists():
        print(f"Erro: A pasta '{folder_path}' não existe.")
        return
    
    if not folder.is_dir():
        print(f"Erro: '{folder_path}' não é uma pasta.")
        return
    
    # Criar pasta de destino
    txt_folder = folder / "txt_files"
    txt_folder.mkdir(exist_ok=True)
    print(f"📁 Pasta criada: {txt_folder}\n")
    
    # Contar ficheiros processados
    count = 0
    
    # Iterar sobre todos os ficheiros da pasta
    for file_path in folder.iterdir():
        # Ignorar a pasta txt_files e subpastas
        if file_path.is_file():
            # Criar nome do ficheiro .txt
            txt_filename = f"{file_path.name}.txt"
            txt_path = txt_folder / txt_filename
            
            try:
                # Copiar o ficheiro para o ficheiro .txt
                shutil.copy2(file_path, txt_path)
                print(f"✓ Copiado: {file_path.name} → {txt_filename}")
                count += 1
            except Exception as e:
                print(f"✗ Erro ao copiar {file_path.name}: {e}")
    
    print(f"\nTotal de ficheiros copiados: {count}")
    print(f"Localização: {txt_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Copia todos os ficheiros de uma pasta para ficheiros .txt individuais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python copy_to_txt.py ./minha_pasta
  python copy_to_txt.py scripts/
  python copy_to_txt.py /caminho/absoluto/pasta
        """
    )
    
    parser.add_argument(
        "folder",
        help="Caminho da pasta a processar"
    )
    
    args = parser.parse_args()
    copy_files_to_txt(args.folder)


if __name__ == "__main__":
    main()