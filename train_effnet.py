
import tensorflow as tf
from models import build_efficientnet_b0, compile_eff
from data_pipeline import train_ds_rgb, val_ds_rgb, test_ds_rgb, class_weight
from utils import get_callbacks, plot_history, evaluate_on_test
from config import ARTIFACTS_DIR, IMG_SIZE

BEST_PATH = f"{ARTIFACTS_DIR}/eff_b0_best.keras"

def main():
    # (opcional) reproducibilidad
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    # Si tu GPU lo soporta bien:
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model, base = build_efficientnet_b0(IMG_SIZE, dropout=0.3)
    compile_eff(model, lr=1e-3)

    cbs = get_callbacks(BEST_PATH, patience=8)

    # Fase 1 (congelada)
    hist1 = model.fit(
        train_ds_rgb, validation_data=val_ds_rgb,
        epochs=15, class_weight=class_weight,
        callbacks=cbs, verbose=1
    )

    # Fase 2 (fine-tuning, sin recargar)
    base.trainable = True
    for l in base.layers[:-50]:
        l.trainable = False
    compile_eff(model, lr=1e-4)

    hist2 = model.fit(
        train_ds_rgb, validation_data=val_ds_rgb,
        epochs=20, class_weight=class_weight,
        callbacks=cbs, verbose=1
    )

    # Cargar mejor y evaluar
    best_eff = tf.keras.models.load_model(BEST_PATH, compile=False)
    compile_eff(best_eff, lr=1e-4)

    test_metrics = best_eff.evaluate(test_ds_rgb, verbose=0)
    print("EffNet test metrics:", dict(zip(best_eff.metrics_names, test_metrics)))
    evaluate_on_test(best_eff, test_ds_rgb, threshold=0.5, name="EffNetB0 (best)")

    # Plots
    plot_history(hist1, "EffNetB0 - Fase 1 (congelada)")
    plot_history(hist2, "EffNetB0 - Fase 2 (fine-tuning)")

    # Guardar copia final en artifacts
    best_eff.save(f"{ARTIFACTS_DIR}/pneumonia_effnet.keras")

if __name__ == "__main__":
    main()
