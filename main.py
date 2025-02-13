from pdf_processor import extract_text_from_pdf
from ner_trainer import train_ner_model
from ner_extractor import extract_entities
from ner_trainer import train_ner_model

# where the validation data is stored
PDF_PATH = "validation_data/test.pdf"
# where the models are saved after the inital train
MODELS_PATH = "models/"
# path where the csv files are
DATA_PATH = "datasets/"
# name of the dataset that will be worked with
# this is also the name of the csv before the .csv extension
SET_NAME = "materials"

# Step 1: Train NER model (Run once, then comment it out)
trained_model = train_ner_model(DATA_PATH, MODELS_PATH, SET_NAME)
# Step 2: Extract text from PDF
text = extract_text_from_pdf(PDF_PATH)
# Step 3: Extract structured data using the trained model
extracted_items = extract_entities(text, trained_model)
# Step 4: Display extracted information
for item in extracted_items:
    print(f"label: {item['label']}, text: {item['text']}")
