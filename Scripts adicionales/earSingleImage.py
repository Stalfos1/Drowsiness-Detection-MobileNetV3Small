import dlib

# Función para detectar y guardar los puntos landmark en un archivo .pts
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
    
    # Creamos el nombre del archivo .pts
    pts_filename = image_path.replace(".jpg", ".pts")  # Cambiar la extensión de la imagen a .pts
    
    # Guardamos los puntos landmark en el archivo .pts
    with open(pts_filename, 'w') as f:
        f.write("version: 1\n")
        f.write("n_points: 68\n")
        f.write("{\n")
        for i in range(68):
            f.write(f"{landmarks.part(i).x} {landmarks.part(i).y}\n")
        f.write("}")

# Ruta de la imagen que deseas procesar
image_path = "example/drowsy/daniNocheLentesTintados_67.jpg"

# Ruta al modelo shape_predictor_68_face_landmarks.dat
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Procesar la imagen
guardar_puntos_landmark(image_path, predictor_path)
