import cv2
import numpy as np
from fisheye_transformation import apply_fisheye

def yolo_to_absolute(bboxes, img_w, img_h):
    """Convierte coordenadas YOLO a píxeles absolutos."""
    abs_boxes = []
    for cls_id, x, y, bw, bh in bboxes:
        x_center = x * img_w
        y_center = y * img_h
        box_w = bw * img_w
        box_h = bh * img_h

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        abs_boxes.append((cls_id, x1, y1, x2, y2))
    return abs_boxes

def generate_bbox_mask(image_shape, bbox_absolute):
    """Genera una máscara individual para cada bounding box."""
    masks = []
    for _, x1, y1, x2, y2 in bbox_absolute:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
        masks.append(mask)
    return masks

def get_fisheye_yolo_bboxes(fisheye_mask, img_w, img_h, circle_center, circle_radius):
    """Obtiene las coordenadas YOLO del bounding box transformado y las ajusta dentro del área visible."""
    rows, cols = np.where(fisheye_mask > 0)
    
    if len(rows) == 0:
        return None  # No se encontraron puntos transformados
    
    # Corrección del intercambio de coordenadas
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)

    # Recortar dentro del área del círculo fisheye
    x_min, x_max = np.clip([x_min, x_max], circle_center[0] - circle_radius, circle_center[0] + circle_radius)
    y_min, y_max = np.clip([y_min, y_max], circle_center[1] - circle_radius, circle_center[1] + circle_radius)

    # Si la *bounding box* está completamente fuera del círculo, la descartamos
    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None

    # Convertir a formato YOLO normalizado
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    w_bbox = (x_max - x_min) / img_w
    h_bbox = (y_max - y_min) / img_h

    return [x_center, y_center, w_bbox, h_bbox]

def draw_yolo_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """Dibuja todas las bounding boxes en la imagen a partir de coordenadas YOLO."""
    h, w = image.shape[:2]
    img_copy = image.copy()

    for cls_id, x, y, bw, bh in bboxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_copy, f"Class {cls_id}", (x1, max(y1 - 5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

def load_yolo_bboxes(txt_path):
    """Carga los bounding boxes en formato YOLO desde un archivo .txt."""
    bboxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            if len(values) == 5:  # Asegurar formato correcto
                bboxes.append(values)  # [class_id, x_center, y_center, width, height]
    return bboxes

def generate_new_bboxes(image_shape, bbox_path, map_x, map_y, output_txt_path):
    """Corrige cada bounding box individualmente en la imagen fisheye."""
    h, w = image_shape[:2]
    
    # Cargar bounding boxes originales
    bboxes_yolo = load_yolo_bboxes(bbox_path)
    
    if not bboxes_yolo:
        print("❌ Error: No se encontraron bounding boxes en el archivo.")
        return None
    
    # Definir el área visible (círculo generado por el efecto fisheye)
    circle_center = (w // 2, h // 2)
    circle_radius = min(circle_center)  # Radio máximo del círculo visible

    # Convertir YOLO a coordenadas absolutas
    bboxes_absolute = yolo_to_absolute(bboxes_yolo, w, h)

    # Generar una máscara individual para cada bounding box
    bbox_masks = generate_bbox_mask(image_shape, bboxes_absolute)

    # Aplicar la transformación fisheye a cada máscara por separado
    transformed_bboxes = []
    
    for i, bbox in enumerate(bboxes_yolo):
        cls_id = int(bbox[0])  # Clase del objeto
        mask = bbox_masks[i]
        fisheye_bbox_mask = apply_fisheye(mask, map_x, map_y)

        new_bbox = get_fisheye_yolo_bboxes(fisheye_bbox_mask, w, h, circle_center, circle_radius)
        if new_bbox:
            transformed_bboxes.append([cls_id] + new_bbox)
        else:
            print(f"⚠ Advertencia: La bounding box {bbox} se salió del área visible y fue eliminada.")

    # Guardar los nuevos bounding boxes en formato YOLO
    with open(output_txt_path, "w") as f:
        for bbox in transformed_bboxes:
            f.write(" ".join(f"{val:.6f}" for val in bbox) + "\n")

    print(f"✅ Bounding boxes transformadas guardadas en: {output_txt_path}")
    return transformed_bboxes
