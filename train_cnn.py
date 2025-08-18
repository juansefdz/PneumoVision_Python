# train_cnn.py
import tensorflow as tf
from models import build_cnn, compile_model
from data_pipeline import train_ds_prep, val_ds_prep, test_ds_prep, class_weight
from utils import get_callbacks, plot_history, evaluate_on_test
from config import ARTIFACTS_DIR

BEST_PATH = f"{ARTIFACTS_DIR}/cnn_best.keras"

def main():
    # (opcional) reproducibilidad
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model = build_cnn()
    compile_model(model, lr=1e-3)

    cbs = get_callbacks(BEST_PATH, patience=8)
    hist = model.fit(
        train_ds_prep, validation_data=val_ds_prep,
        epochs=50, class_weight=class_weight,
        callbacks=cbs, verbose=1
    )

    # Cargar mejor y evaluar
    best_cnn = tf.keras.models.load_model(BEST_PATH, compile=False)
    compile_model(best_cnn, lr=1e-4)

    test_metrics = best_cnn.evaluate(test_ds_prep, verbose=0)
    print("CNN test metrics:", dict(zip(best_cnn.metrics_names, test_metrics)))
    evaluate_on_test(best_cnn, test_ds_prep, threshold=0.5, name="CNN (best)")

    # Plots
    plot_history(hist, "CNN – Historia de entrenamiento")

if __name__ == "__main__":
    main()
