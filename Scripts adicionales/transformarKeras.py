import tensorflow as tf

# Cargar el modelo .keras
model = tf.keras.models.load_model('098 Corregido ear_best_model.keras')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimización para reducir tamaño
tflite_model = converter.convert()

# Guardar el modelo convertido
with open('098_Corregido_ear_best_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Conversión completada. Modelo guardado como 098_Corregido_ear_best_model.tflite")
