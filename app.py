import json
import ocr

from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret key"

@app.route('/', methods=['POST'])
def process_image():
    return json.dumps({
        "hot dog": "$5",
        "ice cream": "$3",
        "soda": "$2"
    })

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
        flash(f'OCRed text: {res['text']}', "success")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost')