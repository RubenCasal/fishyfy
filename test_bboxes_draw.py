from generate_new_bboxes import draw_yolo_bboxes, load_yolo_bboxes
import cv2
image_path = "./input/image5.jpg"
label_path = "./input/label5.txt"
save_result_path = "./output/bbox5.jpg"

bbox = load_yolo_bboxes(label_path)
image = cv2.imread(image_path)
result_img = draw_yolo_bboxes(image, bbox)

cv2.imwrite(save_result_path, result_img)



