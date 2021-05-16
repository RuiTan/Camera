import pytesseract as ocr
from PIL import Image
text = ocr.image_to_string(Image.open("benchmark/number.png"),lang="snum")
print(text)
