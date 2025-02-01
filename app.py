from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image(image_path):
    """ Extracts pixel density along the longer axis of the image """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    intensity = np.mean(img, axis=0 if width >= height else 1)  # Mean pixel density
    return intensity

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    return jsonify({"message": "Image uploaded", "filename": file.filename})

@app.route('/process/<filename>', methods=['GET'])
def process(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    intensity = process_image(file_path)
    return jsonify({"intensity_values": intensity.tolist()})

@app.route('/graph/<filename>', methods=['GET'])
def graph(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    intensity = process_image(file_path)
    
    plt.figure()
    plt.plot(intensity, color="blue")
    plt.xlabel("Distance along longer axis")
    plt.ylabel("Pixel Density")
    plt.title("Densitometry Profile")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
