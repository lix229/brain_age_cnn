import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import argparse
import re


def predict_age(model, image_path):
    """Predict age for a single image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    age_pred = model.predict(img_array, verbose=0)[0][0]
    return age_pred


def process_csv_integration(csv_path, img_dir):
    """
    Process CSV file when it becomes available.
    Expected CSV format: patient_id, age
    """
    df = pd.read_csv(csv_path)
    
    # Create a mapping of patient_id to age
    age_mapping = {}
    for _, row in df.iterrows():
        patient_id = str(row['patient_id']).zfill(4)  # Ensure 4-digit format
        age = float(row['age'])
        age_mapping[patient_id] = age
    
    # Match with existing images
    matched_data = []
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            # Extract patient ID from filename
            match = re.search(r'sub-(\d+)', filename)
            if match:
                patient_id = match.group(1)
                if patient_id in age_mapping:
                    matched_data.append({
                        'filename': filename,
                        'patient_id': patient_id,
                        'age': age_mapping[patient_id]
                    })
    
    return pd.DataFrame(matched_data)


def main(args):
    # Load model
    print(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path, compile=False)
    
    if args.image_path:
        # Single image prediction
        age_pred = predict_age(model, args.image_path)
        print(f"Predicted age for {os.path.basename(args.image_path)}: {age_pred:.1f} years")
    
    elif args.batch_dir:
        # Batch prediction
        results = []
        for filename in sorted(os.listdir(args.batch_dir)):
            if filename.endswith('.jpg'):
                image_path = os.path.join(args.batch_dir, filename)
                age_pred = predict_age(model, image_path)
                
                # Extract patient ID
                match = re.search(r'sub-(\d+)', filename)
                patient_id = match.group(1) if match else 'unknown'
                
                results.append({
                    'filename': filename,
                    'patient_id': patient_id,
                    'predicted_age': age_pred
                })
                print(f"{filename}: {age_pred:.1f} years")
        
        # Save results
        if args.output_csv:
            df = pd.DataFrame(results)
            df.to_csv(args.output_csv, index=False)
            print(f"\nResults saved to {args.output_csv}")
            
            # If ground truth CSV is provided, calculate errors
            if args.truth_csv:
                truth_df = process_csv_integration(args.truth_csv, args.batch_dir)
                merged_df = pd.merge(df, truth_df[['patient_id', 'age']], 
                                   on='patient_id', how='inner')
                merged_df['error'] = merged_df['predicted_age'] - merged_df['age']
                merged_df['abs_error'] = merged_df['error'].abs()
                
                mae = merged_df['abs_error'].mean()
                rmse = np.sqrt((merged_df['error'] ** 2).mean())
                
                print(f"\nEvaluation metrics:")
                print(f"MAE: {mae:.2f} years")
                print(f"RMSE: {rmse:.2f} years")
                
                # Save detailed results
                output_detailed = args.output_csv.replace('.csv', '_detailed.csv')
                merged_df.to_csv(output_detailed, index=False)
                print(f"Detailed results saved to {output_detailed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain age prediction inference (Keras)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--batch_dir', type=str, help='Directory with images for batch prediction')
    parser.add_argument('--output_csv', type=str, help='Path to save predictions')
    parser.add_argument('--truth_csv', type=str, help='Path to CSV with true ages for evaluation')
    
    args = parser.parse_args()
    main(args)