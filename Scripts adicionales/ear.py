import dlib
import os

# Función para detectar y guardar los puntos landmark en un archivo .dot
def guardar_puntos_landmark(image_path, predictor_path):
    # Cargamos el detector de puntos landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Leemos la imagen
    img = dlib.load_rgb_image(image_path)
    
    # Detectamos caras en la imagen
    faces = detector(img)
    
    # Si no se detecta ninguna cara, salimos de la función
    if len(faces) == 0:
        print(f"No se encontraron caras en la imagen: {image_path}")
        return
    
    # Obtenemos los puntos landmark para la primera cara detectada
    landmarks = predictor(img, faces[0])
    
    # Creamos el nombre del archivo .dot
    dot_filename = os.path.splitext(image_path)[0] + ".pts"
    
    # Guardamos los puntos landmark en el archivo .dot
    with open(dot_filename, 'w') as f:
        f.write("version: 1\n")
        f.write("n_points: 68\n")
        f.write("{\n")
        for i in range(68):
            f.write(f"{landmarks.part(i).x} {landmarks.part(i).y}\n")
        f.write("}")

# Directorio donde se encuentran las imágenes
input_folder = "example/drowsy"

# Ruta al modelo shape_predictor_68_face_landmarks.dat
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Iteramos sobre todas las imágenes en el directorio
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filtro para imágenes
        image_path = os.path.join(input_folder, filename)
        guardar_puntos_landmark(image_path, predictor_path)
