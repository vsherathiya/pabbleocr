import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import re
import csv
import numpy as np
from PIL import Image, ImageEnhance  # Import Pillow

# Initialize PaddleOCR
ocr_model = PaddleOCR(lang='en', use_gpu=False)

image_path = r'D:\Data\3-bhk-1755-sq-ft.jpg'

# Initialize darkness factor
darkness_factor = 1.28

# Read the image with OpenCV
image = cv2.imread(image_path)

# Convert the OpenCV image to a Pillow image
pillow_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Create an ImageEnhance object
enhancer = ImageEnhance.Brightness(pillow_image)

# Darken the image
darkened_pillow_image = enhancer.enhance(darkness_factor)

# Convert the darkened Pillow image back to a NumPy array
darkened_image = cv2.cvtColor(np.array(darkened_pillow_image), cv2.COLOR_RGB2BGR)

# Perform text extraction
result = ocr_model.ocr(darkened_image)

# Extracting detected components
boxes = [res[0] for res in result[0]]
texts = [res[1][0] for res in result[0]]
scores = [res[1][1] for res in result[0]]

# Import our image
# image = cv2.imread(image_path)

# Define the pattern to avoid
avoid_pattern = r'^\d{4}[\sxX*]*\d{4}$'

# Visualize our image and detections
# plt.figure(figsize=(15, 15))

# Set a threshold for drop_score (you can adjust this threshold)
# drop_score = 0.5

def calculate_area(dimensions):
    # feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?"?\s?(\d{1,2})[\'\"*]?[xX*]?[-.]?\s?(\d{1,2})[\'\"*]?'
    feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?"?\s?[xX*]?\s?(\d{1,2})[\'\"*]?\s?[*-.]?\s?(\d{1,2})[\'\"*]?'
    # feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?\s?[xX*]?[-.]?\s?(\d{1,2})[\'\"]?'

    match = re.match(feet_inches_pattern, dimensions)

    if match:
        feet1, inches1, feet2, inches2 = map(int, match.groups())
        total_inches = (feet1 * 12 + inches1) * (feet2 * 12 + inches2)
        sqft = total_inches / 144.0
        sqyd = sqft * 0.111111
        return sqyd
    else:
        return None

# Initialize variables to store total square yards
total_sqyd = 0.0

# Draw annotations on image and process text
for i in range(len(boxes)):
    box = boxes[i]
    text = texts[i]
    score = scores[i]
    
    # Check if the text matches the avoid_pattern
    if not re.match(avoid_pattern, text):
        # Draw bounding box
        cv2.rectangle(image, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 255, 0), 2)
        
        # Position the text slightly below the bounding box to avoid overlap
        text_position = (int(box[0][0]), int(box[2][1]) + 20)
        
        # Increase font size for other text
        font_scale =  1.1
        
        # Draw text near the bounding box
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        # Match pattern and calculate square
        feet_inches_pattern = r'(\d{1,2})[\'\"*]?\s?[-.]?\s?(\d{1,2})[\'\"*]?"?\s?[xX*]?\s?(\d{1,2})[\'\"*]?\s?[*-.]?\s?(\d{1,2})[\'\"*]?'


        match = re.match(feet_inches_pattern, text)
        
        if match:
            dimensions = match.group(0)
            sqyd = calculate_area(dimensions)
            
            if sqyd is not None:
                cv2.putText(image, f'SqYd: {sqyd:.2f}', (int(box[0][0]), int(box[2][1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                total_sqyd += sqyd

# # Check if resizing is required based on the width
# desired_width = 1000
# image_height, image_width, _ = image.shape
# resize_ratio = desired_width / image_width
# print(resize_ratio)
# if resize_ratio < 1.0:
#     # Resize the image to a consistent size
#     resized_image = cv2.resize(image, (desired_width, int(image_height * resize_ratio)))
# else:
#     resized_image = image

# Save the annotated image
annotated_image_path = 'annotated_image.jpg'
cv2.imwrite(annotated_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# Create a CSV file for the extracted data
output_csv_path = 'output.csv'

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header row
    csvwriter.writerow(['Text', 'Score', 'Bounding Box', 'Square Yards'])
    
    # Write data rows
    for i in range(len(boxes)):
        text = texts[i]
        score = scores[i]
        box = ','.join(map(str, boxes[i]))
        
        # Check if the text matches the avoid_pattern
        if not re.match(avoid_pattern, text):
            match = re.match(feet_inches_pattern, text)
            sqyd = calculate_area(text) if match else None
            csvwriter.writerow([text, score, box, sqyd])

# Add total square yards to the image
# cv2.rectangle(resized_image, (0, 0), (150, 80), (0, 0, 0), -1)
cv2.putText(image, f'Total SqYd: {total_sqyd:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

print(f"Total Square Yards: {total_sqyd:.2f}")
print(f"OCR results saved to {output_csv_path}")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
