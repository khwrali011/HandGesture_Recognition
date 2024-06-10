# from flask import Flask, request, render_template, send_file
# from ultralytics import YOLO
# import cv2
# import os
# import io
# from PIL import Image
# import torch

# app = Flask(__name__)

# # Load the model
# model = YOLO(r"weights\YOLOV8_HG_Detection.pt")

# def predict(img_path):
#     # Perform prediction
#     results = model.predict(img_path)

#     # Extract boxes, probabilities, and class ids
#     boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
#     # boxes = boxes[0]
#     probs = results[0].boxes.conf if results[0].boxes.conf is not None else None  # Probabilities
#     if probs:
#         probs = probs[0]
#     class_id = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None  # Class IDs
#     if class_id:
#         class_id = class_id[0]

#     print(boxes)
#     print(len(boxes))
#     print(probs)
#     print(class_id)

#     # Read the image
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Draw boxes on the image
#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             img = cv2.putText(img, f"{class_id}: {probs:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
#     # Convert image to PIL format
#     img = Image.fromarray(img)
#     return img

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return "No file part"
#         file = request.files['file']
#         if file.filename == '':
#             return "No selected file"
#         if file:
#             # Save the file to a temporary location
#             img_path = os.path.join('static', file.filename)
#             file.save(img_path)

#             # Perform prediction
#             img = predict(img_path)

#             # Save the resulting image
#             result_path = os.path.join('static', 'result.png')
#             img.save(result_path)

#             return send_file(result_path, mimetype='image/png')

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, render_template, send_file, redirect, url_for
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import torch
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:abcd@127.0.0.1:3306/yolo_predictions'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Ensure the static directory exists
if not os.path.exists('./static'):
    os.makedirs('./static')

# Load the model
model = YOLO(r"weights\YOLOV8_HG_Detection.pt")

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    boxes = db.Column(db.Text, nullable=True)
    probabilities = db.Column(db.Text, nullable=True)
    class_ids = db.Column(db.Text, nullable=True)
    result_image_path = db.Column(db.String(255), nullable=False)

def predict(img_path):
    # Perform prediction
    results = model.predict(img_path)
    
    # Extract boxes, probabilities, and class ids
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results[0].boxes.xyxy is not None else []
    probs = results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes.conf is not None else []
    class_ids = results[0].boxes.cls.cpu().numpy().tolist() if results[0].boxes.cls is not None else []

    # Read the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw boxes on the image
    for box, prob, class_id in zip(boxes, probs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(class_id)}: {prob:.2f}"
        img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert image to PIL format
    img = Image.fromarray(img)
    return img, boxes, probs, class_ids

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the file to a temporary location
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

            # Perform prediction
            img, boxes, probs, class_ids = predict(img_path)

            # Save the resulting image
            result_path = os.path.join('static', f"result_{file.filename}")
            img.save(result_path)

            # Save to database
            prediction = Prediction(
                filename=file.filename,
                boxes=str(boxes),
                probabilities=str(probs),
                class_ids=str(class_ids),
                result_image_path=result_path
            )
            db.session.add(prediction)
            db.session.commit()

            return send_file(result_path, mimetype='image/png')

    return render_template('index.html')

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
