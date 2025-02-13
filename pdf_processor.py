import pytesseract
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path, dpi=300):
    """Converts PDF to images and extracts text using Tesseract OCR."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    text = "\n".join([pytesseract.image_to_string(page) for page in pages])
    return text
