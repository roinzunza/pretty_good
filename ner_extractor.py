import spacy

def load_ner_model(model_path="custom_ner_model"):
    """Loads the trained NER model."""
    return spacy.load(model_path)

def extract_entities(text, model_path="custom_ner_model"):
    """Extracts structured data (Quantity, Material, Weight) from text."""
    nlp_custom = load_ner_model(model_path)
    doc = nlp_custom(text)

    extracted_items = []
    for ent in doc.ents:
        if ent.label_ in ["QUANTITY","MATERIAL", "TYPE", "HEIGHT_WIDTH", "THICKNESS"]:
            extracted_items.append({"text": ent.text, "label": ent.label_})

    return extracted_items
