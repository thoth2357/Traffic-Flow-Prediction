import re
import csv

rtf_content = open("datasat.rtf", "r").read()

# Extract headers
headers = re.findall(r"\b(\w+)\s+", rtf_content.split("\\par")[0])

# Extract data lines and remove '\par'
data_lines = re.findall(r"\b\d+\b[\s\S]+?\\par", rtf_content)
data_lines = [re.sub(r'\\par|(?<=\d)\s(?=\d)', '', line) for line in data_lines]

# Write to CSV
csv_filename = "output.csv"
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(headers)  # Write headers
    for line in data_lines:
        csv_writer.writerow(re.findall(r"\b\d+\b|[\d.-]+|[A-Za-z\s]+", line))  # Write data lines

print(f"Conversion completed. CSV file saved as {csv_filename}")
