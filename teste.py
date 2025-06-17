from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

img = Image.open("teste.jpg")
texto = pytesseract.image_to_string(img, lang='por')
print(texto)
