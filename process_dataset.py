import os
import cv2
import numpy as np
import yaml
from fisheye_transformation import create_LUT_table, apply_fisheye, resize_to_square
from generate_new_bboxes import generate_new_bboxes

def process_split(input_dir, output_dir, map_x, map_y, output_shape):

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

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: the image can not be loaded{filename}")
                continue

            # Resize image
            image = resize_to_square(image)

            # Apply LUT
            fisheye_img = apply_fisheye(image, map_x, map_y)

            # New bounding boxes
            output_label_path = os.path.join(output_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            generate_new_bboxes(image.shape, label_path, map_x, map_y, output_label_path)
            # transform to gray image
            #fisheye_img = cv2.cvtColor(fisheye_img, cv2.COLOR_BGR2GRAY)
            
            # Resize to desired dimensions
            fisheye_img = cv2.resize(fisheye_img, output_shape, interpolation=cv2.INTER_AREA)
        
            output_image_path = os.path.join(output_images, filename)
            cv2.imwrite(output_image_path, fisheye_img)

            print(f"Processed: {filename}")

    print(f"Fisheye transformation completed: {input_dir}")

def update_yaml(input_yaml, output_yaml, output_dir):

    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    
    data['train'] = os.path.join(output_dir, "train")
    data['val'] = os.path.join(output_dir, "valid")
    if 'test' in data:
        data['test'] = os.path.join(output_dir, "test")

    # save new yaml
    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("✅ Archivo data.yaml actualizado.")

def process_roboflow_dataset(input_dir, output_dir, distortion_strength, output_shape):

    os.makedirs(output_dir, exist_ok=True)

    # Get the first image shape to create LUT
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
        print("Error: Any image has been found in the dataset.")
        return

    # Load the first image to create LUT
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print("Error: The image can not be loaded.")
        return

    first_image = resize_to_square(first_image)
    map_x, map_y = create_LUT_table(first_image,distortion_strength)

    # Process the data for every split (train, valid, test) 
    for split in ["train", "valid", "test"]:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        if os.path.exists(input_split_dir):
            process_split(input_split_dir, output_split_dir, map_x, map_y, output_shape)

    # Update data.yaml
    input_yaml = os.path.join(input_dir, "data.yaml")
    output_yaml = os.path.join(output_dir, "data.yaml")
    update_yaml(input_yaml, output_yaml, output_dir)
