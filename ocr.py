import re
from typing import Dict, Optional, Tuple
from pytesseract import Output, pytesseract
import io
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocess_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Preprocess the image for OCR."""

    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    image = result_norm

    # Resize the image
    rescaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_OTSU)

    # Denoise the image using morphological operations (opening)
    #kernel = np.ones((2,2),np.uint8)
    #denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    denoised_image = binary_image

    # Adjust contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_adjusted_image = clahe.apply(denoised_image)
    #contrast_adjusted_image = denoised_image
    inverted_image = cv2.bitwise_not(contrast_adjusted_image)
    
    processed_img = inverted_image

    return processed_img

def ocr_image(image_stream: io.BytesIO, return_img = None) -> Tuple[Dict[str, str], Optional[cv2.typing.MatLike]]:
    """OCR the image data and return the result as JSON."""

    image_stream.seek(0)
    img_np = np.frombuffer(image_stream.read(), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    processed_img = preprocess_image(img)
    data = pytesseract.image_to_data(processed_img, output_type=Output.DICT, config='--psm 4')

    return data, image_resize(processed_img, height=500) if return_img else None

def isolate_prices(data: Dict[str, str]) -> Dict[str, str]:
    """Isolate the prices from OCR data
    1. transform to lines based off of empty ''
    2. match regex for rows with prices
    """
    
    lines = []
    line = []
    for i in range(len(data['text'])):
        if data['text'][i] == '':
            lines.append(line)
            line = []
        else:
            line.append(data['text'][i])
    lines.append(line)
    
    res = {}
    pattern = r'([0-9]+\.[0-9]+)'
    for line in lines:
        matches = re.search(pattern, ' '.join(line))
        if matches:
            price= matches.group(0)
            for i, word in enumerate(line):
                if price in word:
                    res[' '.join(line[:i])] = float(price)
                    break
    return res

if __name__ == "__main__":
    with open("images/text-custom-font.png", "rb") as f:
        data, img = ocr_image(f, True)
        print(data['text'])
        res = isolate_prices(data)
        print(res)
        cv2.imshow("img", img)
        cv2.waitKey(0)
