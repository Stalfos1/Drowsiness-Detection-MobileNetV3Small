import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=8):
    # Abre el video
    video_capture = cv2.VideoCapture(video_path)
    
    # Obtiene la información del video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Verifica si el FPS es válido
    if fps <= 0:
        print("Error: El video tiene un FPS inválido.")
        return
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcula la cantidad de fotogramas a saltar para obtener 8 fotogramas por segundo
    frame_skip = int(round(fps / frame_rate))
    
    # Lee y guarda los fotogramas
    count = 0
    success = True
    while success:
        success, image = video_capture.read()
        if count % frame_skip == 0:
            if success:
                # Guarda el fotograma en el directorio de salida
                frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{count // frame_skip + 1}.jpg"
                output_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(output_path, image)
        count += 1

    video_capture.release()

if __name__ == "__main__":
    # Ruta del video de entrada
    video_path = "../Propios/alexLentesTintados.mp4"
    
    # Directorio de salida para los fotogramas
    output_directory = "fotogramas"
    
    # Crea el directorio de salida si no existe
    os.makedirs(output_directory, exist_ok=True)
    
    # Extrae los fotogramas del video
    extract_frames(video_path, output_directory)
