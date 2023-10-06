import os
import re
import cv2
from paddleocr import PaddleOCR
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_gpu=False)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with a strong secret key

# Define the square yard area calculation function
def calculate_area(dimensions):
    feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?"?\s?[xX*]?\s?(\d{1,2})[\'\"*]?\s?[*-.]?\s?(\d{1,2})[\'\"*]?'

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
# Function to process the uploaded image
def process_image(file_path):
    try:
        # Load the original image
        image = cv2.imread(file_path)
        image = cv2.resize(image, (2600, 1600))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to create a binary image
        # You can adjust the block size and C value as needed
        binary_image = cv2.adaptiveThreshold(
            gray, 256   , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1
        )

        # Perform image enhancement (you may need to define 'darkened_pillow_image' here)
        # Example:
        enhanced_pillow_image = Image.fromarray(binary_image)
        enhancer = ImageEnhance.Brightness(enhanced_pillow_image)
        darkness_factor = 1.27
        darkened_pillow_image = enhancer.enhance(darkness_factor)
        # The `darkness_factor` and other image enhancement code remains the same
        

        # Convert the darkened Pillow image back to a NumPy array
        darkened_image = cv2.cvtColor(np.array(darkened_pillow_image), cv2.COLOR_RGB2BGR)

        # Perform text extraction with PaddleOCR
        result = ocr.ocr(darkened_image)


        # Initialize a list to store extracted data and regions
        extracted_data = []

        # Define a regex pattern to match dimensions like "11'-0" x 13'-0" and square yard areas like "200 sq. yd"
        pattern = r'(\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?"\s?[xX*]\s?\d{1,2}[\'\"]?\s?[-.]?\s?\d{1,2}[\'\"]?)|(\d+\.?\d*)\s?sq\.?\s?yd'

        # Define an avoidance pattern to skip certain patterns during extraction
        avoid_pattern = r'^\d{4}\s?[*xX]\s?\d{4}$'

        # Iterate through the results of text extraction
        for detection in result[0]:
            text = detection[1][0]
            bbox = np.array(detection[0])

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

        # Calculate the total square yard area
        total_sqyd_area = round(sum(data['sqyd_area'] for data in extracted_data if data.get('sqyd_area') is not None), 2)
        total_sqyd_text = f"Total Square Yard Area: {round(total_sqyd_area)} sqyd (approx.)"
        output_image = darkened_image.copy()
        # Create a copy of the enhanced image for adding the total area text
        output_image_with_total_area = output_image.copy()

        # Calculate the backg size for the total area text
        (total_area_text_width, total_area_text_height), _ = cv2.getTextSize(total_sqyd_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
        total_area_backg_width = total_area_text_width + 40  # Add some padding
        total_area_backg_height = total_area_text_height + 20
        total_area_backg_color = (0, 0, 0)  # Black backg color
        total_area_font_color = (255, 255, 255)  # White text color

        # Calculate the coordinates for the total area backg
        total_area_x1 = 20  # Adjust as needed for horizontal placement
        total_area_y1 = 20  # Adjust as needed for vertical placement
        total_area_x2 = total_area_x1 + total_area_backg_width
        total_area_y2 = total_area_y1 + total_area_backg_height

        # Draw the total area backg
        cv2.rectangle(output_image_with_total_area, (total_area_x1, total_area_y1), (total_area_x2, total_area_y2), total_area_backg_color, -1)

        # Calculate the coordinates for the total area text
        total_area_text_x = total_area_x1 + 10  # Adjust for horizontal placement within the backg
        total_area_text_y = total_area_y1 + 40  # Adjust for vertical placement within the backg

        # Draw the total area text
        cv2.putText(output_image_with_total_area, total_sqyd_text, (total_area_text_x, total_area_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, total_area_font_color, 5, cv2.LINE_AA, False)

        # Save the output image
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
        cv2.imwrite(output_image_path, output_image_with_total_area)

        return output_image_path, extracted_data
    except Exception as e:
        # Handle exceptions, log errors, and return an error message
        print(f"Error processing image: {str(e)}")
        return None, []


# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Save the uploaded image to the upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Process the uploaded image
            processed_image, extracted_data = process_image(filename)
            print(extracted_data)
            df = pd.DataFrame(extracted_data)
            print(df)
            df.to_csv('new.csv')
            if processed_image:
                # Calculate the total square yard area
                total_sqyd_area = sum(data['sqyd_area'] for data in extracted_data if data.get('sqyd_area') is not None)
                return render_template('result.html', uploaded_image_url=filename, masked_image_url=processed_image,
                                       matched_dimensions=extracted_data, total_sqyd_area=(total_sqyd_area))
            else:
                return render_template('error.html', message='Error processing the image.')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
