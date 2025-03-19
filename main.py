from process_dataset import process_roboflow_dataset


INPUT_DATASET = "./Face Detection dataset"
OUTPUT_DATASET = "./Fisheye Face Detection dataset"
DISTORTION_STRENGTH = 0.7 # [0-1] value


process_roboflow_dataset(INPUT_DATASET, OUTPUT_DATASET, DISTORTION_STRENGTH)
