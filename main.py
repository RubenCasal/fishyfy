from process_dataset import process_roboflow_dataset


INPUT_DATASET = "./Face Detection dataset"
OUTPUT_DATASET = "./Fisheye Face Detection dataset"
DISTORTION_STRENGTH = 0.6  # [0-1] value
OUTPUT_SHAPE = (848, 800) # Desired dimensions for the images in the final dataset

process_roboflow_dataset(INPUT_DATASET, OUTPUT_DATASET, DISTORTION_STRENGTH, OUTPUT_SHAPE)
