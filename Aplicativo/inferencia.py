import cv2
import numpy as np
import tensorflow as tf
import time
import dlib
from collections import deque

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='098_Corregido_ear_best_model.tflite')
interpreter.allocate_tensors()

# Obtener detalles de los tensores de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configuración de PERCLOS
WINDOW_DURATION = 10  # Ventana de tiempo en segundos para PERCLOS
PERCLOS_THRESHOLD = 0.3  # Umbral para activar alarma

# Cargar el detector de rostros de dlib y el predictor de landmarks
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Función para calcular el EAR (Eye Aspect Ratio)
def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Variables para PERCLOS y FPS
drowsy_frames = deque(maxlen=WINDOW_DURATION * 30)  # 30 FPS aprox.
frame_times = deque(maxlen=10)  # Últimos 10 tiempos de fotogramas

def detect_somnolence():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        inference_time = 0
        start_time = time.time()

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Dibujar los ojos
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            
            # Preprocesar la imagen para el modelo
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (224, 224)) / 255.0
            face_resized = np.expand_dims(face_resized, axis=0).astype(np.float32)
            
            # Tiempo antes de la inferencia
            inference_start = time.time()

            # Ejecutar modelo TFLite
            interpreter.set_tensor(input_details[0]['index'], face_resized)
            interpreter.set_tensor(input_details[1]['index'], np.array([[ear]], dtype=np.float32))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            inference_end = time.time()
            inference_time = inference_end - inference_start
            
            predicted_label = np.argmax(predictions)
            label = "Drowsy" if predicted_label == 0 else "Non-Drowsy"
            color = (0, 0, 255) if predicted_label == 0 else (255, 0, 0)
            
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Guardar resultado en la ventana de PERCLOS
            drowsy_frames.append(1 if predicted_label == 0 else 0)
            perclos = sum(drowsy_frames) / len(drowsy_frames) if len(drowsy_frames) > 0 else 0
            
            if perclos > PERCLOS_THRESHOLD:
                print('ALERTA: Somnolencia detectada por PERCLOS')
                time.sleep(1)

        end_time = time.time()
        frame_time = end_time - start_time
        frame_times.append(frame_time)
        avg_fps = len(frame_times) / sum(frame_times) if len(frame_times) > 1 else 0

        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Infer: {inference_time:.4f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Detección de Somnolencia con PERCLOS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_somnolence()