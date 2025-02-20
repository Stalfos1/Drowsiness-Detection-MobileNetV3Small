import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
import pygame
import time
from collections import deque

# Cargar el modelo previamente entrenado
model = load_model('098 Corregido ear_best_model.keras')

# Configuración de PERCLOS
WINDOW_DURATION = 5  # Ventana de tiempo en segundos para PERCLOS
PERCLOS_THRESHOLD = 0.3  # Umbral para activar alarma

# Función para calcular el EAR
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Cargar el detector de rostros y el predictor de landmarks de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar pygame para la alarma
pygame.mixer.init()
pygame.mixer.music.load('alarm-clock.mp3')

# Variables para el cálculo de PERCLOS
drowsy_frames = deque(maxlen=WINDOW_DURATION * 30)  # 30 FPS aprox.

# Función principal para detectar somnolencia con PERCLOS basado en el modelo
def detect_somnolence():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Obtener landmarks del rostro
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Dibujar landmarks de los ojos (más pequeños)
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)

            # Calcular EAR de ambos ojos
            left_ear = calculate_ear(np.array(left_eye))
            right_ear = calculate_ear(np.array(right_eye))

            # Ajustar el EAR según el ojo más cerrado
            ear = min(left_ear, right_ear)

            # Preprocesar la imagen para el modelo
            face_roi = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (224, 224)) / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # Crear la entrada EAR en formato esperado
            ear_input = np.array([[ear]], dtype=np.float32)

            # Obtener la predicción del modelo (imagen + EAR)
            predictions = model.predict([face_resized, ear_input])
            predicted_label = np.argmax(predictions)

            # Determinar el color y texto de la etiqueta
            if predicted_label == 0:
                label = "Drowsy"
                color = (0, 0, 255)  # Rojo 🔴
            else:
                label = "Non-Drowsy"
                color = (255, 0, 0)  # Azul 🔵

            # Mostrar el estado de somnolencia en pantalla con el color correspondiente
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Guardar resultado en la ventana de PERCLOS
            drowsy_frames.append(1 if predicted_label == 0 else 0)

            # Calcular PERCLOS (porcentaje de frames "Drowsy" en la ventana)
            perclos = sum(drowsy_frames) / len(drowsy_frames) if len(drowsy_frames) > 0 else 0

            # Activar alarma si PERCLOS supera el umbral
            if perclos > PERCLOS_THRESHOLD:
                print('ALERTA: Somnolencia detectada por PERCLOS')
                pygame.mixer.music.play(-1, 0.0)
                time.sleep(3)
                pygame.mixer.music.stop()

        cv2.imshow("Detección de Somnolencia con PERCLOS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para detectar somnolencia
detect_somnolence()
