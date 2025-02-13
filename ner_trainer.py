import spacy
import csv
import random
import datetime
import os
from spacy.training import Example

# Generate dynamic timestamp
TIMESTAMP = datetime.datetime.now().strftime("%m-%d-%y-%H")


def load_training_data(csv_file):
    """Reads training data from a CSV file and converts it to spaCy format dynamically."""
    training_data = []

    with open(csv_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames  # Extract column names as labels

        for row in reader:
            text = " ".join(row.values())  # Concatenate all column values
            annotations = {"entities": []}

            for label in headers:
                entity_text = row[label]
                start_idx = text.find(entity_text)
                if start_idx != -1:
                    annotations["entities"].append(
                        (start_idx, start_idx + len(entity_text), label.upper())
                    )

            training_data.append((text, annotations))

    return training_data, [h.upper() for h in headers]


def train_ner_model(data_path, model_path, set_name):
    """Trains a spaCy Named Entity Recognition (NER) model using CSV data with dynamic labels."""

    csv_file = f"{data_path}{set_name}.csv"

    # Load training data and dynamically determine entity labels
    TRAIN_DATA, labels = load_training_data(csv_file)

    print(f"Training data: {set_name} \n\tlabels: {labels}")

    # Create model name with timestamp
    model_name = f"{model_path}{set_name}_{TIMESTAMP}"

    # Check if a model already exists
    if os.path.exists(model_name):
        print(f"Loading existing model from {model_name}")
        nlp = spacy.load(model_name)
        ner = nlp.get_pipe("ner")
        optimizer = nlp.resume_training()  # Continue training the model
    else:
        print("No existing model found. Creating a new model.")
        nlp = spacy.load("en_core_web_sm")
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
            optimizer = nlp.initialize()

    # Add any new labels found in the dataset
    for label in labels:
        if label not in ner.labels:
            ner.add_label(label)

    for i in range(75):  # Training iterations
        random.shuffle(TRAIN_DATA)
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, sgd=optimizer)

    # Save model
    nlp.to_disk(model_name)
    print(f"Model saved to {model_name}")
    return model_name


def load_model(model_path):
    """Load the existing NER model."""
    nlp = spacy.load(model_path)
    return nlp
