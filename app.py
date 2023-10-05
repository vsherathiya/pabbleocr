import os
import re
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageEnhance
import pandas as pd
import csv
from paddleocr import PaddleOCR

app = Flask(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en', use_gpu=False)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with a strong secret key

# Define the square yard area calculation function
def calculate_area(dimensions):
    feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?"?\s?(\d{1,2})[\'\"*]?\s?[*-.]?\s?(\d{1,2})[\'\"*]?'

    match = re.match(feet_inches_pattern, dimensions)

    if match:
        feet1, inches1, feet2, inches2 = map(int, match.groups())
        total_inches = (feet1 * 12 + inches1) * (feet2 * 12 + inches2)
        sqft = total_inches / 144.0
        sqyd = sqft * 0.111111
        return sqyd
    else:
        return None

# Function to process the uploaded image
def process_image(file_path):
    try:
        # Load the original image
        image = cv2.imread(file_path)

        # Convert the OpenCV image to a Pillow image
        pillow_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Create an ImageEnhance object
        enhancer = ImageEnhance.Brightness(pillow_image)

        # Darken the image
        darkened_pillow_image = enhancer.enhance(1.28)

        # Convert the darkened Pillow image back to a NumPy array
        darkened_image = cv2.cvtColor(np.array(darkened_pillow_image), cv2.COLOR_RGB2BGR)

        # Perform OCR with PaddleOCR
        result = ocr.ocr(darkened_image)

        # Initialize a list to store extracted data and regions
        extracted_data = []

        # Define a regex pattern to match dimensions and square yard areas
        pattern = r'(\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?"\s?[xX*]\s?\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?)|(\d+\.?\d*)\s?sq\.?\s?yd'

        # Define an avoidance pattern to skip certain patterns during extraction
        avoid_pattern = r'^\d{4}\s?[*xX]\s?\d{4}$'

        # Iterate through the results of text extraction
        for line in result[0]:
            text = line[1].strip()
            bbox = line[0]

            # Replace 'A' with '4' for area calculation
            text = text.replace('A', '4')
            text = text.replace('o', '0')
            text = text.replace('O', '0')

            # Check if the text matches the desired pattern using regex
            match = re.search(pattern, text)

            if match:
                # If the match is a dimension pattern
                if match.group(1):
                    # Calculate the square yard area
                    dimensions = re.findall(r'(\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?"\s?[xX*]\s?\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?)', text)
                    area = 1.0  # Default value
                    if dimensions:
                        # Extract the dimensions and calculate the area
                        area = calculate_area(dimensions[0])

                    extracted_data.append({'text': text, 'bbox': bbox, 'sqyd_area': area})

                # If the match is a square yard area pattern
                elif match.group(2):
                    sqyd_area = float(match.group(2))
                    extracted_data.append({'text': text, 'bbox': bbox, 'sqyd_area': sqyd_area})

            # Check if the text matches the avoidance pattern and skip it
            elif re.search(avoid_pattern, text):
                pass
            else:
                extracted_data.append({'text': text, 'bbox': bbox})

        # Create a copy of the original image for drawing boxes and text annotations
        output_image = image.copy()

        for data in extracted_data:
            bbox = data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            text_to_write = data['text']

            # Calculate the background size to match the text
            (text_width, text_height), _ = cv2.getTextSize(text_to_write, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            backg_width = text_width + 10  # Add some padding
            backg_height = text_height + 10
            backg_color = (255, 255, 255)  # White background color
            font_color = (0, 0, 255)  # Red text color for square yard area
            cv2.rectangle(output_image, (x1, y1 - backg_height), (x1 + backg_width, y1), backg_color, -1)
            cv2.putText(output_image, text_to_write, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2, cv2.LINE_AA, False)

        # Draw boxes and text for square yard area above the original text
        for data in extracted_data:
            bbox = data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            text_to_write = data['text']

            if data.get('sqyd_area') is not None:
                # Calculate the background size to match the text
                (text_width, text_height), _ = cv2.getTextSize(text_to_write, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                backg_width = text_width + 5  # Add some padding
                backg_height = text_height + 10
                backg_color = (255, 255, 255)
                font_color = (0, 0, 255)  # Red text color

                # Calculate the coordinates for the rectangle
                rectangle_x1 = x1
                rectangle_y1 = y2  # Place the rectangle below the text
                rectangle_x2 = x1 + backg_width
                rectangle_y2 = y2 + backg_height

                # Draw the rectangle
                cv2.rectangle(output_image, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), backg_color, -1)

                # Draw the text
                cv2.putText(output_image, text_to_write, (x1, y2 + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2,
                            cv2.LINE_AA, False)

        # Save the output image with annotations
        annotated_image_path = os.path.join(UPLOAD_FOLDER, 'annotated_image.jpg')
        cv2.imwrite(annotated_image_path, output_image)

        # Save extracted data to a CSV file
        csv_path = os.path.join(UPLOAD_FOLDER, 'extracted_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Text', 'Bounding Box', 'Square Yards']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in extracted_data:
                writer.writerow({'Text': data['text'], 'Bounding Box': data['bbox'], 'Square Yards': data.get('sqyd_area', '')})

        return annotated_image_path, csv_path

    except Exception as e:
        return None, str(e)

# Helper function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for uploading images and processing
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Generate a random filename to prevent overwriting
            import uuid
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded image
            annotated_image_path, csv_path = process_image(file_path)

            if annotated_image_path and csv_path:
                return render_template('result.html', image_path=annotated_image_path, csv_path=csv_path)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
