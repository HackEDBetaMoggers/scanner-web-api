from typing import Dict
from pytesseract import Output, pytesseract
import io
import cv2
import numpy as np

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
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Denoise the image using morphological operations (opening)
    #kernel = np.ones((2,2),np.uint8)
    #denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    denoised_image = binary_image

    # Adjust contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_adjusted_image = clahe.apply(denoised_image)
    #contrast_adjusted_image = denoised_image
    
    processed_img = contrast_adjusted_image

    return processed_img

def ocr_image(image_stream: io.BytesIO) -> Dict[str, str]:
    """OCR the image data and return the result as JSON."""

    image_stream.seek(0)
    img_np = np.frombuffer(image_stream.read(), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    processed_img = preprocess_image(img)

    data = pytesseract.image_to_data(processed_img, output_type=Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return data, image_resize(processed_img, height=500)

def isolate_prices(data: Dict[str, str]) -> Dict[str, str]:
    """Isolate the prices from the OCR data."""
    """
    algo:
        1. isolate dollar signs
        2. look to the left until another dollar sign is encountered to get the title
    """
    dollar_indices = [i for i, x in enumerate(data['text']) if '$' in x]
    for i in dollar_indices[::-1]:
        j = i
        item = []
        while j > 0 and '$' not in data['text'][j]:
            j -= 1
            item.append(data['text'][j])
        print(data['text'][i])

if __name__ == "__main__":
    with open("images/receipt1_1.jpg", "rb") as f:
        data, img = ocr_image(f)
        print(data['text'])
        isolate_prices(data)
        cv2.imshow("img", img)
        cv2.waitKey(0)