import pandas as pd
import pydicom
import os
from PIL import Image
from tqdm import tqdm
import sys

# Define paths relative to the script location
csv_path = 'Train_Labels.csv'
dicom_dir = 'src/Train_Images/' 
output_image_dir = 'dataset/images/'
output_label_dir = 'dataset/labels/'

# Create output directories if they don't exist
print("Creating output directories...")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Read the CSV
print(f"Reading CSV from {csv_path}...")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"FATAL ERROR: Could not find the CSV file at {os.path.abspath(csv_path)}")
    print("Please make sure 'Train_Labels.csv' is in the same folder as this script.")
    sys.exit()


print(f"Found {len(df)} total entries.")
print(f"Processing {df['patientId'].nunique()} unique patients.")

# Add a counter for found files
files_found = 0

# Group by patientId to handle multiple boxes per image
for patientId, rows in tqdm(df.groupby('patientId'), desc="Processing Patients"):
    # Load DICOM file
    dcm_path = os.path.join(dicom_dir, f'{patientId}.dcm')
    
    try:
        # Check if file exists *before* trying to read it
        if not os.path.exists(dcm_path):
            # This will skip files that are in the CSV but not in the image folder
            # This is normal for this dataset
            continue 
            
        dcm = pydicom.dcmread(dcm_path)
        image_pixels = dcm.pixel_array
        
        files_found += 1

        # DICOM pixel values can be inverted, check Photometric Interpretation
        if 'PhotometricInterpretation' in dcm and dcm.PhotometricInterpretation == 'MONOCHROME1':
            image_pixels = image_pixels.max() - image_pixels

        # Normalize to 0-255 and convert to image
        max_pixel = image_pixels.max()
        if max_pixel > 0:
            image_pixels = (image_pixels / max_pixel * 255).astype('uint8')
        else:
            image_pixels = image_pixels.astype('uint8')
            
        img = Image.fromarray(image_pixels)
        
        # Save image as PNG
        img.save(os.path.join(output_image_dir, f'{patientId}.png'))
        
        # Check if there's a detection (Target=1) and process bounding boxes
        if rows['Target'].iloc[0] == 1:
            with open(os.path.join(output_label_dir, f'{patientId}.txt'), 'w') as label_file:
                for index, row in rows.iterrows():
                    img_w, img_h = img.size
                    
                    if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['width']) and pd.notna(row['height']):
                        x, y, w, h = row['x'], row['y'], row['width'], row['height']
                        
                        dw = 1. / img_w
                        dh = 1. / img_h
                        center_x = (x + w / 2.0) * dw
                        center_y = (y + h / 2.0) * dh
                        norm_w = w * dw
                        norm_h = h * dh
                        
                        label_file.write(f'0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n')

    except FileNotFoundError:
        # This block will now be skipped because of the 'if not os.path.exists' check above
        pass
        
    except Exception as e:
        print(f"Error processing {patientId}: {e}")

if files_found == 0:
    print("\nWARNING: Preprocessing completed but found 0 matching files.")
else:
    print(f"\nPreprocessing complete! Successfully processed {files_found} images.")