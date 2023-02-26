import cv2
import pytesseract
import pandas as pd
import os

# Set up camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Create dataframe to store license plate numbers
df = pd.DataFrame(columns=['License Plate'])

# Capture images until a license plate is detected
while True:
    # Read image from camera
    ret, frame = camera.read()

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plate using cascade classifier
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Loop through detected plates
    for (x, y, w, h) in plates:
        # Draw rectangle around plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop license plate region
        plate_img = gray[y:y + h, x:x + w]

        # Apply threshold to license plate region
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply dilation to license plate region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        plate_img = cv2.dilate(plate_img, kernel, iterations=1)

        # Get license plate number using Tesseract
        plate_number = pytesseract.image_to_string(plate_img, lang='eng', config='--psm 11')

        # Clean up license plate number
        plate_number = plate_number.strip().replace(' ', '').replace('\n', '').replace('\r', '')

        # Add license plate number to dataframe if it's not empty
        if plate_number != '':
            df = df.append({'License Plate': plate_number}, ignore_index=True)

            # Save image of license plate
            cv2.imwrite(os.path.join('images', plate_number + '.jpg'), plate_img)

            # Show license plate number
            print(plate_number)

            # Exit loop
            break

    # Show frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()

# Show license plate numbers
print(df)
