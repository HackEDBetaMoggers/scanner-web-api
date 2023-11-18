import os
from typing import Dict
import pytesseract
import io
import PIL as Image
from PIL import Image, ImageOps, ImageFilter

from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret key"


def ocr_image(image_stream: io.BytesIO) -> Dict[str, str]:
    """OCR the image data and return the result as JSON."""
    image = Image.open(image_stream)
    image = ImageOps.grayscale(image)
    text = pytesseract.image_to_string(image, lang="eng")
    return {"text": text}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files["img_name"]
        if image.filename == "":
            flash('Please Upload Image file', "danger")
            return redirect(request.url)
        filename = secure_filename(image.filename)
        image.save(f"./images/{filename}")
        res = ocr_image(image.stream)
        flash('File upload Successfully !', "success")
        flash(f'OCRed text: {res['text']}', "success")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost')