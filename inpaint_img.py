import cv2
import numpy as np
import easyocr

image = cv2.imread("test_image.png")

reader = easyocr.Reader(["ch_sim", "en"])
results = reader.readtext(image)

mask = np.zeros(image.shape[:2], dtype=np.uint8)

for bbox, text, prob in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    x_min = int(min(top_left[0], bottom_left[0]))
    y_min = int(min(top_left[1], top_right[1]))
    x_max = int(max(bottom_right[0], top_right[0]))
    y_max = int(max(bottom_right[1], bottom_left[1]))

    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=cv2.FILLED)

inpainted = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

cv2.imshow("Text Mask", mask)
cv2.imshow("Inpainted Image", inpainted)
cv2.imwrite("text_mask.jpg", mask)
cv2.imwrite("output.jpg", inpainted)
