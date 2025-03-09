from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
from autocorrect import Speller
import os, easyocr
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import math


def perform_ocr(image_path, reader):
    # Perform OCR on the image
    # result = reader.readtext(image_path, width_ths = 0.8,  decoder = 'wordbeamsearch')
    result = ocr.ocr(image_path, cls=True)

    # Extract text and bounding boxes from the OCR result

    extracted_text_boxes = []
    for line in result:
        extracted_text_boxes.append((line[0][0], line[0][1][0]))
    # extracted_text_boxes = [(entry[0], entry[1]) for entry in result if entry[2] > 0.1]
    print(extracted_text_boxes)

    return extracted_text_boxes


def get_font(image, text, width, height):

    # Default values at start
    # font_size = None  # For font size
    font_size = int(height / 1.2)
    print(font_size)
    font = None  # For object truetype with correct font size
    box = None  # For version 8.0.0
    x = 0
    y = 0

    draw = ImageDraw.Draw(image)  # Create a draw object

    # Test for different font sizes
    for size in range(1, 500):

        # Create new font
        # new_font = ImageFont.load_default(size=font_size)
        new_font = ImageFont.truetype(
            "/home/zy/Project/video/script/python-image-translator/HanyiSentyPagoda Regular.ttf",
            font_size,
        )

        # Calculate bbox for version 8.0.0
        new_box = draw.textbbox((0, 0), text, font=new_font)

        # Calculate width and height
        new_w = new_box[2] - new_box[0]  # Bottom - Top
        new_h = new_box[3] - new_box[1]  # Right - Left

        # If too big then exit with previous values
        if new_w > width or new_h > height:
            break

        # Set new current values as current values
        font_size = size
        font = new_font
        box = new_box
        w = new_w
        h = new_h

        # Calculate position (minus margins in box)
        x = (width - w) // 2 - box[0]  # Minus left margin
        y = (height - h) // 2 - box[1]  # Minus top margin

    return font, x, y


def add_discoloration(color, strength):
    # Adjust RGB values to add discoloration
    r, g, b, a = color
    r = max(0, min(255, r + strength))  # Ensure RGB values are within valid range
    g = max(0, min(255, g + strength))
    b = max(0, min(255, b + strength))

    if r == 255 and g == 255 and b == 255:
        r, g, b = 245, 245, 245

    return (r, g, b)


def get_background_color(image, x_min, y_min, x_max, y_max):
    # Define the margin for the edges
    margin = 10

    # Crop a small region around the edges of the bounding box
    edge_region = image.crop(
        (
            max(x_min - margin, 0),
            max(y_min - margin, 0),
            min(x_max + margin, image.width),
            min(y_max + margin, image.height),
        )
    )

    # Find the most common color in the cropped region
    edge_colors = edge_region.getcolors(edge_region.size[0] * edge_region.size[1])
    background_color = max(edge_colors, key=lambda x: x[0])[1]
    if len(background_color) == 3:  # If it's RGB, add an alpha value
        background_color = (*background_color, 255)

    # Add a bit of discoloration to the background color
    background_color = add_discoloration(background_color, 40)

    return background_color


def get_text_fill_color(background_color):
    # Calculate the luminance of the background color
    luminance = (
        0.299 * background_color[0]
        + 0.587 * background_color[1]
        + 0.114 * background_color[2]
    ) / 255

    # Determine the text color based on the background luminance
    if luminance > 0.5:
        return "black"  # Use black text for light backgrounds
    else:
        return "white"  # Use white text for dark backgrounds


def midpoint(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    The start point will be the mid-point between the top-left corner and
    the bottom-left corner of the box.
    the end point will be the mid-point between the top-right corner and the bottom-right corner.

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return x_mid, y_mid


import cv2
import numpy as np
import math
from PIL import Image


def midpoint(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


import cv2
import numpy as np
import math
from PIL import Image


def midpoint(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def cv_remove_text(file_from: str, file_to: str) -> None:
    """
    Remove text from an image.

    :param file_from: Input image path
    :param file_to: Output image path
    :return: None
    """
    print(file_from, file_to, sep=" -> ")

    # Load image using PIL (in RGB format)
    pil_img = Image.open(file_from)
    image = np.array(pil_img)  # Keep it in RGB

    # Convert image to BGR (OpenCV processes in BGR)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure mask is 8-bit 1-channel
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # OCR Detection
    result = ocr.ocr(file_from, cls=True)
    if not result or result[0] is None:
        print("No text detected")
        return
    print(result)

    for item in result:
        box = item[0][0]
        x0, y0 = map(int, box[0])
        x1, y1 = map(int, box[1])
        x2, y2 = map(int, box[2])
        x3, y3 = map(int, box[3])

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1)) ** 2)
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)

    inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    # Save output
    cv2.imwrite(file_to, inpainted_img)


def replace_text_with_translation(image_path, translated_texts, text_boxes):
    # Open the image with OpenCV
    image_cv = cv2.imread(image_path)
    image_pil = Image.open(image_path).convert("RGBA")

    for text_box, translated in zip(text_boxes, translated_texts):
        if translated is None:
            continue

        # Get text bounding box coordinates
        x_min, y_min = min([p[0] for p in text_box[0]]), min(
            [p[1] for p in text_box[0]]
        )
        x_max, y_max = max([p[0] for p in text_box[0]]), max(
            [p[1] for p in text_box[0]]
        )

        # Convert to integers
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Extract the region of interest (ROI)
        roi = image_cv[y_min:y_max, x_min:x_max]

        # Get average background color
        avg_color = np.median(roi, axis=(0, 1))

        # Create a solid color mask
        solid_bg = np.full_like(roi, avg_color, dtype=np.uint8)

        # Apply Gaussian Blur (dynamic kernel size))
        blur_size = max(25, ((x_max - x_min) // 5) * 2 + 1)  # Ensure odd number
        blurred_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 30)

        # Blend the blurred region with the solid background color
        blended_roi = cv2.addWeighted(blurred_roi, 0.7, solid_bg, 0.3, 0)

        # Replace the original text area with the blended version
        # image_cv[y_min:y_max, x_min:x_max] = blended_roi

    # Convert OpenCV image back to PIL
    image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # Replace each text box with translated text
    for text_box, translated in zip(text_boxes, translated_texts):
        if translated is None:
            continue

        x_min, y_min = min([p[0] for p in text_box[0]]), min(
            [p[1] for p in text_box[0]]
        )
        x_max, y_max = max([p[0] for p in text_box[0]]), max(
            [p[1] for p in text_box[0]]
        )

        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Calculate font size
        font, text_x, text_y = get_font(image, translated, x_max - x_min, y_max - y_min)

        # Get text color that contrasts with background
        text_color = get_text_fill_color(avg_color)

        # Draw the translated text
        draw.text(
            (x_min + text_x, y_min + text_y),
            translated,
            fill=text_color,
            font=font,
            encoding="utf-8",
        )

    return image


speller = Speller(lang="en")

# Initialize the OCR reader
reader = easyocr.Reader(["ch_sim", "en"], model_storage_directory="model")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Initialize the Translator
translator = GoogleTranslator(source="en", target="zh-CN")

# Define input and output location
input_folder = "input"
output_folder = "output"

# Process each image file from input
files = os.listdir(input_folder)
image_files = [file for file in files if file.endswith((".jpg", ".jpeg", ".png"))]
for filename in image_files:

    print(f"[INFO] Processing {filename}...")

    image_path = os.path.join(input_folder, filename)
    image_path_no_text = output_folder + "/no_text_" + filename

    cv_remove_text(image_path, image_path_no_text)

    # Extract text and location
    extracted_text_boxes = perform_ocr(image_path, ocr)

    # Translate texts
    translated_texts = []
    for text_box, text in extracted_text_boxes:
        translated_texts.append(translator.translate(text))
    print(translated_texts)

    # Replace text with translated text
    image = replace_text_with_translation(
        image_path_no_text, translated_texts, extracted_text_boxes
    )

    # Save modified image
    base_filename, extension = os.path.splitext(filename)
    output_filename = f"{base_filename}-translated{extension}"
    output_path = os.path.join(output_folder, output_filename)
    image.save(output_path)

    print(f"[INFO] Saved as {output_filename}...")
