import cv2
import numpy as np

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load the accessory images (PNG with transparent background)
glasses = cv2.imread("effect_img/glasses.png", -1)  # Glasses
mask = cv2.imread("effect_img/mask.png", -1)  # Mask
hat = cv2.imread("effect_img/hat.png", -1)  # Hat


def add_accessory_to_face(frame, face, accessory_img, position, scale=1):
    # Get the face region dimensions
    face_x, face_y, face_w, face_h = face

    # Resize the accessory image based on the face size
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)
    accessory_resized = cv2.resize(accessory_img, (349, 349))

    # Get the coordinates to position the accessory
    x_offset = position[0]
    y_offset = position[1]

    # Ensure the accessory image has an alpha channel (transparency)
    if (
        accessory_resized.shape[2] == 4
    ):  # If the accessory has transparency (alpha channel)
        # Split the accessory image into BGR and Alpha channels
        accessory_bgr = accessory_resized[:, :, :3]
        accessory_alpha = accessory_resized[:, :, 3] / 255.0

        # Get the face region in the original frame
        face_region = frame[
            face_y + y_offset : face_y + y_offset + new_h,
            face_x + x_offset : face_x + x_offset + new_w,
        ]

        # Perform alpha blending to add the accessory on top of the face region
        for c in range(3):  # Process each color channel
            face_region[:, :, c] = (1 - accessory_alpha) * face_region[
                :, :, c
            ] + accessory_alpha * accessory_bgr[:, :, c]

        # Update the frame with the new face region
        frame[
            face_y + y_offset : face_y + y_offset + new_h,
            face_x + x_offset : face_x + x_offset + new_w,
        ] = face_region
    else:
        # If the accessory doesn't have transparency, just place it on top of the face region
        frame[
            face_y + y_offset : face_y + y_offset + new_h,
            face_x + x_offset : face_x + x_offset + new_w,
        ] = accessory_resized


# Load and display the image
frame = cv2.imread("human.png")

# Convert to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Add accessories to the detected faces
for x, y, w, h in faces:
    # Add glasses
    add_accessory_to_face(frame, (x, y, w, h), glasses, (349, 349), scale=0.5)

    # Add mask
    # add_accessory_to_face(frame, (x, y, w, h), mask, (int(w * 0.1), int(h * 0.5)), scale=1.0)

    # Add hat
    # add_accessory_to_face(frame, (x, y, w, h), hat, (int(w * 0.25), -int(h * 0.25)), scale=0.6)

# Display the decorated image
cv2.imshow("Decorated Faces", frame)

# Save the result
cv2.imwrite("output_image.jpg", frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
