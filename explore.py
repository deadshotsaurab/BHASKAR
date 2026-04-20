import pandas as pd
import os

folder = 'c:/Users/saura/OneDrive/Desktop/bhaskar/ROC_curve'
file_path = os.path.join(folder, 'training_data.xlsx')

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
    print("Info:\n", df.info())
else:
    print(f"File {file_path} not found.")

csv_path = os.path.join(folder, 'training_data.csv')
if os.path.exists(csv_path):
    print("Found training_data.csv!!")
