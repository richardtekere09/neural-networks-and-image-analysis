# import cv2
# import sys
# import pytesseract

# if __name__ == '__main__':

#   if len(sys.argv) < 2:
#     print('Usage: python ocr_simple.py image.jpg')
#     sys.exit(1)
  
#   # Read image path from command line
#   imPath = sys.argv[1]
    
#   # Uncomment the line below to provide path to tesseract manually
#   # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#   # Define config parameters.
#   # '-l eng'  for using the English language
#   # '--oem 1' sets the OCR Engine Mode to LSTM only.
#   #
#   #  There are four OCR Engine Mode (oem) available
#   #  0    Legacy engine only.
#   #  1    Neural nets LSTM engine only.
#   #  2    Legacy + LSTM engines.
#   #  3    Default, based on what is available.
#   #
#   #  '--psm 3' sets the Page Segmentation Mode (psm) to auto.
#   #  Other important psm modes will be discussed in a future post.  


#   config = ('-l eng --oem 1 --psm 3')

#   # Read image from disk
#   im = cv2.imread(imPath, cv2.IMREAD_COLOR)

#   # Run tesseract OCR on image
#   text = pytesseract.image_to_string(im, config=config)

#   # Print recognized text
#   print(text)

import sys
from pathlib import Path
import cv2
import pytesseract

# If tesseract isn't on PATH, uncomment and set the correct path:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Apple Silicon
# pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"     # Intel Macs


def load_image(path_str: str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Could not read image (unsupported format?): {p}")
    return im, p


def preprocess_for_ocr(bgr):
    # Light, generic preprocessing: grayscale -> slight blur -> adaptive threshold
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # Adaptive threshold helps on variable lighting docs; comment out if it harms your images
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return bin_img


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_words.py <image_path>")
        sys.exit(1)

    im, in_path = load_image(sys.argv[1])

    # Choose the image you want Tesseract to see:
    #  - try binarized for documents; switch to 'im' if it hurts natural scenes
    ocr_input = preprocess_for_ocr(im)

    # Tesseract config:
    #  -l eng  : English language
    # --oem 1  : LSTM engine
    # --psm 3  : Fully automatic page segmentation
    config = "-l eng --oem 1 --psm 3"

    # image_to_data gives bounding boxes + text + confidence per element
    data = pytesseract.image_to_data(
        ocr_input, config=config, output_type=pytesseract.Output.DICT
    )

    # Draw boxes and collect results
    boxed = im.copy()
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf_str = data["conf"][i]
        try:
            conf = float(conf_str)
        except ValueError:
            conf = -1.0

        if text and conf > 0:  # filter out blanks and invalid lines
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # optionally put tiny labels (comment if cluttered)
            # cv2.putText(boxed, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            words.append((text, conf, (x, y, w, h)))

    # Print recognized words in reading order Tesseract gives
    print("=== Recognized words (confidence %) ===")
    for t, c, _ in words:
        print(f"{t}\t[{c:.1f}]")

    # Save annotated image next to input
    out_path = in_path.with_suffix("").as_posix() + "_boxed.jpg"
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), boxed)
    print(f"\nSaved annotated image with boxes to: {out_path}")

    # If you also want the full plain text:
    full_text = pytesseract.image_to_string(ocr_input, config=config)
    print("\n=== Full text ===")
    print(full_text)


if __name__ == "__main__":
    main()
