import os
import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_config():
    with open("configs/params.yaml") as f:
        return yaml.safe_load(f)

def preprocess_image(img_path, target_size):
    """Preprocess a single image"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image {img_path}")
    
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0  # Normalize to [0,1]

def process_samples(input_dir, output_dir, target_size, num_samples=4):
    """Process and save only a few sample images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ['tumor', 'no_tumor']:
        class_dir = os.path.join(input_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping")
            continue
        
        print(f"\nProcessing {num_samples} {class_name} samples...")
        
        # Get first few image files
        image_files = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        # Prepare data storage
        csv_data = []
        pdf_images = []
        
        for img_name in image_files:
            try:
                img_path = os.path.join(class_dir, img_name)
                img = preprocess_image(img_path, target_size)
                
                # Store for CSV
                csv_data.append({
                    'filename': img_name,
                    'class': class_name,
                    'pixels': img.flatten().tolist()  # Flatten image data
                })
                
                # Store for PDF
                pdf_images.append((img, img_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Save CSV
        if csv_data:
            csv_path = os.path.join(output_dir, f"{class_name}_samples.csv")
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)
            print(f"Saved {len(csv_data)} samples to {csv_path}")
        
        # Save PDF
        if pdf_images:
            pdf_path = os.path.join(output_dir, f"{class_name}_samples.pdf")
            with PdfPages(pdf_path) as pdf:
                for img, img_name in pdf_images:
                    plt.figure(figsize=(5,5))
                    plt.imshow(img)
                    plt.title(f"{class_name}: {img_name}")
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
            print(f"Saved PDF report to {pdf_path}")

def main():
    config = load_config()
    
    # Setup output directory
    output_dir = "results/sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing sample images...")
    
    # Process training data samples
    process_samples(
        config['train_dir'],
        output_dir,
        tuple(config['input_shape'][:2]),
        num_samples=4  # Change this number if you want more/less samples
    )
    
    print("\nSample processing complete!")
    print(f"Output saved to: {output_dir}")

if __name__ == '__main__':
    main()