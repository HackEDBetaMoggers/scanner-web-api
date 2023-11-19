import io
import json

import cv2
import ocr
import base64
import codecs

from flask import Flask, render_template, request, redirect, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"

@app.route('/process', methods=['POST'])
def process_image():
    image = json.loads(request.data).get("image")
    if not image:
        return "No image provided", 400
    image = image[image.find(",")+1:]
    
    binary_img = io.BytesIO(base64.decodebytes(codecs.encode(image, 'utf-8')))
    binary_img.seek(0)
    data, _ = ocr.ocr_image(binary_img, True)
    res = ocr.isolate_prices(data)
    response = jsonify(res)
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files["img_name"]
        if image.filename == "":
            flash('Please Upload Image file', "danger")
            return redirect(request.url)
        filename = secure_filename(image.filename)
        image.save(f"./images/{filename}")
        res = ocr.ocr_image(image.stream)
        flash(f"OCRed text: {res['text']}", "success")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')