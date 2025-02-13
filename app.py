import csv

TRAIN_DATA = [
    ("3 A500 Steel Square Tubing - 2 x 2 x 3/16", {
        "entities": [
            (0, 1, "QUANTITY"),
            (2, 12, "MATERIAL"),
            (13, 26, "TYPE"),
            (29, 34, "HEIGHT_WIDTH"),
            (37, 41, "THICKNESS"),
        ]
    }),
    ("25 HSS 2 x 2 x 3/16 (ASTM A500) 4.32", {
        "entities": [
            (0, 2, "QUANTITY"),
            (21, 30, "MATERIAL"),
            (3, 6, "TYPE"),
            (15, 19, "HEIGHT_WIDTH"),
        ]
    }),
    # Add more items here...
]

# Define CSV header
csv_headers = ["QUANTITY", "MATERIAL", "TYPE", "HEIGHT_WIDTH", "THICKNESS"]

# Process data
csv_data = []
for text, annotations in TRAIN_DATA:
    row = {key: "" for key in csv_headers}  # Initialize row with empty values
    
    for start, end, label in annotations["entities"]:
        row[label] = text[start:end]  # Extract text using the indexes
    
    csv_data.append(row)

# Write to CSV
with open("material_data.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    writer.writerows(csv_data)

print("CSV file created: material_data.csv")
