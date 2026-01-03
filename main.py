# main.py
import argparse
import sys
import os

# Importamos nuestros módulos refactorizados
import config
from preprocess import prepare_dataset

def main():
    parser = argparse.ArgumentParser(description="PneumoVision AI System Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Comando: setup (solo preparar datos)
    subparsers.add_parser("setup", help="Preprocesa el dataset (Letterbox resize)")
    
    # Comando: train
    train_parser = subparsers.add_parser("train", help="Entrenar un modelo")
    train_parser.add_argument("--model", type=str, default="custom", choices=["custom", "effnet"], help="Modelo a usar")
    
    # Comando: predict
    pred_parser = subparsers.add_parser("predict", help="Analizar una imagen")
    pred_parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen")
    pred_parser.add_argument("--model_path", type=str, default=None, help="Ruta manual al .keras (opcional)")
    pred_parser.add_argument("--type", type=str, default="custom", choices=["custom", "effnet"], help="Tipo de modelo usado")

    args = parser.parse_args()

    # 1. Verificación de Datos Automática
    # Si intentamos entrenar, primero aseguramos que los datos existan
    if args.command in ["train", "setup"]:
        success = prepare_dataset()
        if not success and args.command == "train":
            sys.exit(1)
        if args.command == "setup":
            print("Setup finalizado.")
            return

    # 2. Ejecución de Rutinas
    if args.command == "train":
        from train import run_training
        run_training(model_name=args.model)

    elif args.command == "predict":
        from predict import predict_and_explain
        
        # Definir ruta por defecto si no se da
        if args.model_path is None:
            # Asume nombre estándar generado por train.py
            args.model_path = os.path.join(config.ARTIFACTS_DIR, f"{args.type}_best.keras")
            
        if not os.path.exists(args.model_path):
            print(f"❌ Error: No se encontró el modelo en {args.model_path}")
            print("   Entrena primero usando: python main.py train --model custom")
            return

        predict_and_explain(args.model_path, args.image, model_type=args.type)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()