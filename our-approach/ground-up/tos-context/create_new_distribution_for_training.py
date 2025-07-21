import csv
import random
import os
import sys

# Increase the CSV field size limit to maximum for 32-bit long (Windows compatibility)
csv.field_size_limit(2147483647)

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths relative to the script's directory
input_path = os.path.join(script_dir, 'train.tsv')
output_path = os.path.join(script_dir, 'train_unskewed.tsv')

# Read the TSV file with UTF-8 encoding
with open(input_path, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile, delimiter='\t', quotechar='"')
    rows = list(reader)

# Assume first row is header
header = rows[0] if rows else None
data_rows = rows[1:] if rows else []

# Separate rows by label (assuming labels are '0' or '1' after reading)
zero_rows = [row for row in data_rows if row[2] == '0']
one_rows = [row for row in data_rows if row[2] == '1']

num_zero = len(zero_rows)
num_one = len(one_rows)

# Determine the minimum count for balancing (undersample majority)
min_count = min(num_zero, num_one)

# Downsample the majority class
if num_zero > num_one:
    zero_rows = random.sample(zero_rows, min_count)
elif num_one > num_zero:
    one_rows = random.sample(one_rows, min_count)

# Combine and shuffle the balanced rows
balanced = zero_rows + one_rows
random.shuffle(balanced)

# Write to the output TSV file with all fields quoted and UTF-8 encoding
with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
    if header:
        writer.writerow(header)
    writer.writerows(balanced)