import os
import cv2
import numpy as np
import yaml
from fisheye_transformation import create_LUT_table, apply_fisheye, resize_to_square
from generate_new_bboxes import generate_new_bboxes

def process_split(input_dir, output_dir, map_x, map_y):
    """
    Procesa un split del dataset (train, valid, test), aplicando transformación fisheye con una LUT precomputada.

    Parámetros:
    - input_dir: Directorio que contiene "images" y "labels".
    - output_dir: Directorio donde se guardarán las imágenes y etiquetas transformadas.
    - map_x, map_y: LUTs precomputados para la transformación fisheye.
    """

    # Crear carpetas de salida
    output_images = os.path.join(output_dir, "images")
    output_labels = os.path.join(output_dir, "labels")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    input_images = os.path.join(input_dir, "images")
    input_labels = os.path.join(input_dir, "labels")

    for filename in os.listdir(input_images):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(input_images, filename)
            label_path = os.path.join(input_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {filename}")
                continue

            # Redimensionar a cuadrado
            image = resize_to_square(image)

            # Aplicar fisheye con la LUT precomputada
            fisheye_img = apply_fisheye(image, map_x, map_y)

            # Generar nuevas bounding boxes
            output_label_path = os.path.join(output_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            generate_new_bboxes(image.shape, label_path, map_x, map_y, output_label_path)

            # Guardar imagen transformada
            output_image_path = os.path.join(output_images, filename)
            cv2.imwrite(output_image_path, fisheye_img)

            print(f"Procesado: {filename}")

    print(f"✅ Transformación completa para: {input_dir}")

def update_yaml(input_yaml, output_yaml, output_dir):
    """
    Actualiza el archivo data.yaml con las nuevas rutas del dataset transformado.

    Parámetros:
    - input_yaml: Ruta del data.yaml original.
    - output_yaml: Ruta donde se guardará el nuevo data.yaml.
    - output_dir: Directorio base del nuevo dataset transformado.
    """
    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    # Actualizar rutas
    data['train'] = os.path.join(output_dir, "train")
    data['val'] = os.path.join(output_dir, "valid")
    if 'test' in data:
        data['test'] = os.path.join(output_dir, "test")

    # Guardar nuevo archivo yaml
    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("✅ Archivo data.yaml actualizado.")

def process_roboflow_dataset(input_dir, output_dir, distortion_strength):
    """
    Procesa todo un dataset de Roboflow (train, valid, test) aplicando fisheye con LUT precomputada.
    
    Parámetros:
    - input_dir: Directorio del dataset original.
    - output_dir: Directorio donde se guardará el dataset transformado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Obtener el tamaño de la primera imagen del dataset para generar la LUT
    first_image_path = None
    for split in ["train", "valid", "test"]:
        split_images_dir = os.path.join(input_dir, split, "images")
        if os.path.exists(split_images_dir):
            for filename in os.listdir(split_images_dir):
                if filename.endswith((".jpg", ".png")):
                    first_image_path = os.path.join(split_images_dir, filename)
                    break
        if first_image_path:
            break

    if first_image_path is None:
        print("Error: No se encontró ninguna imagen en el dataset.")
        return

    # Cargar la primera imagen para generar la LUT
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print("Error: No se pudo cargar la imagen de referencia para generar la LUT.")
        return

    first_image = resize_to_square(first_image)
    map_x, map_y = create_LUT_table(first_image,distortion_strength)

    # Procesar cada split (train, valid, test) utilizando la LUT precomputada
    for split in ["train", "valid", "test"]:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        if os.path.exists(input_split_dir):
            process_split(input_split_dir, output_split_dir, map_x, map_y)

    # Actualizar data.yaml
    input_yaml = os.path.join(input_dir, "data.yaml")
    output_yaml = os.path.join(output_dir, "data.yaml")
    update_yaml(input_yaml, output_yaml, output_dir)
