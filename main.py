from process_dataset import process_roboflow_dataset


INPUT_DATASET = "./Face Detection dataset"
OUTPUT_DATASET = "./Fisheye Face Detection dataset"

# Ejecutar el procesamiento
process_roboflow_dataset(INPUT_DATASET, OUTPUT_DATASET)
