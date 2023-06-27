from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pytesseract import image_to_string
from PIL import Image
import os
import cv2
import numpy as np
from pdf2image import convert_from_path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/mr.johnson/Documents/Fun Projects/ImageScanner/imagescanner/uploads'

def preprocess_image(image_path):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return

    # Load the image from file
    image = cv2.imread(image_path, 0)

    # Check if the image has been loaded correctly
    if image is None:
        print(f"Failed to load image file: {image_path}")
        return

    # Apply a slight Gaussian blur
    blur = cv2.GaussianBlur(image, (5,5), 0)

    # Perform adaptive thresholding
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Dilation and Erosion
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(threshold, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

    # Deskewing
    coords = np.column_stack(np.where(img_erosion > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save the processed image temporarily, overwrite the original image
    cv2.imwrite(image_path, deskewed)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Apply preprocessing to the image
        preprocess_image(file_path)

        if filename.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            result = ""
            for i in range(len(images)):
                # Convert PIL Image to cv2 Image
                open_cv_image = np.array(images[i]) 
                # Convert RGB to BGR 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 
                # Preprocess image
                preprocess_image(open_cv_image)
                result += image_to_string(open_cv_image, lang='eng', config='--psm 3')
        else:
            result = image_to_string(Image.open(file_path), lang='eng', config='--psm 3')
        return render_template('results.html', result=result)
    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)

