"""
Prepare NOC embeddings for semantic search
This script processes the NOC data and creates embeddings for fast semantic matching
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

print("🔄 Loading NOC data...")
df = pd.read_csv('noc_data_full.csv')

# Parse all list fields from string to list
# The data is stored as pipe-separated strings in CSV
def parse_list_field(x):
    if pd.isna(x) or x == '' or x == 'nan':
        return []
    x_str = str(x).strip()
    if x_str.startswith('['):
        # Try to eval as list
        try:
            return eval(x_str)
        except:
            return []
    elif '|' in x_str:
        # Split by pipe and clean
        return [item.strip() for item in x_str.split('|') if item.strip()]
    else:
        return []

df['main_duties_list'] = df['main_duties'].apply(parse_list_field)
df['example_titles_list'] = df['example_titles'].apply(parse_list_field)
df['exclusions_list'] = df['exclusions'].apply(parse_list_field)

# Create weighted searchable text - focus on main duties but include all fields
print("📝 Creating weighted searchable text for each NOC...")
def create_searchable_text(row):
    parts = []
    
    # Title - weight 2x (repeat twice for higher importance)
    parts.append(f"Title: {row['title']} {row['title']}")
    
    # Description - weight 1.5x
    parts.append(f"Description: {row['description']}")
    parts.append(row['description'][:200])  # Partial repeat for weight
    
    # Main duties - weight 3x (HIGHEST priority - repeat 3 times)
    if row['main_duties_list']:
        main_duties_text = " ".join(row['main_duties_list'])
        parts.append(f"Main duties: {main_duties_text}")
        parts.append(f"Responsibilities: {main_duties_text}")  # Synonym for matching
        parts.append(f"Key duties: {main_duties_text}")  # Additional weight
    
    # Example titles - weight 1x
    if row['example_titles_list']:
        parts.append("Example titles: " + " ".join(row['example_titles_list']))
    
    # Employment requirements - weight 1x
    if pd.notna(row['employment_requirements']) and row['employment_requirements']:
        parts.append(f"Requirements: {row['employment_requirements']}")
    
    # Additional information - weight 0.5x
    if pd.notna(row['additional_information']) and row['additional_information']:
        parts.append(str(row['additional_information'])[:100])  # Partial for lower weight
    
    # Exclusions - weight 0.5x (lower priority)
    if row['exclusions_list']:
        parts.append("Exclusions: " + " ".join(row['exclusions_list'][:3]))  # First 3 only
    
    # Hierarchy info - weight 1x
    if pd.notna(row['broad_category']) and row['broad_category']:
        parts.append(f"Category: {row['broad_category']}")
    if pd.notna(row['major_group']) and row['major_group']:
        parts.append(f"Group: {row['major_group']}")
    
    return " ".join(filter(None, parts))

df['searchable_text'] = df.apply(create_searchable_text, axis=1)

# Load embedding model - upgraded for better accuracy
print("🤖 Loading Sentence Transformer model (this may take a minute)...")
model = SentenceTransformer('all-mpnet-base-v2')  # Higher accuracy model for semantic understanding

# Generate embeddings for full NOC profiles
print(f"🧮 Generating profile embeddings for {len(df)} NOC codes...")
embeddings = model.encode(
    df['searchable_text'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

# Generate individual duty embeddings for duty-by-duty matching
print(f"🎯 Generating duty-level embeddings for precise matching...")
all_duties = []
duty_to_noc_map = []  # Track which NOC each duty belongs to

for idx, row in df.iterrows():
    if row['main_duties_list']:
        for duty in row['main_duties_list']:
            if duty and duty.strip():
                all_duties.append(duty)
                duty_to_noc_map.append(idx)

duty_embeddings = model.encode(
    all_duties,
    show_progress_bar=True,
    batch_size=32
)

print(f"   ✓ Created {len(all_duties)} individual duty embeddings")

# Save embeddings and processed data
print("💾 Saving embeddings and processed data...")
np.save('noc_embeddings.npy', embeddings)
np.save('duty_embeddings.npy', duty_embeddings)

# Save metadata with all fields
metadata = {
    'noc_codes': df['noc_code'].tolist(),
    'titles': df['title'].tolist(),
    'descriptions': df['description'].tolist(),
    'main_duties': df['main_duties_list'].tolist(),
    'example_titles': df['example_titles_list'].tolist(),
    'employment_requirements': df['employment_requirements'].tolist(),
    'additional_information': df['additional_information'].tolist(),
    'exclusions': df['exclusions_list'].tolist(),
    'urls': df['url'].tolist(),
    'searchable_texts': df['searchable_text'].tolist(),
    'all_duties': all_duties,
    'duty_to_noc_map': duty_to_noc_map
}

with open('noc_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Successfully prepared embeddings for {len(df)} NOC codes!")
print(f"   Profile embedding shape: {embeddings.shape}")
print(f"   Duty embedding shape: {duty_embeddings.shape}")
print(f"   Files created:")
print(f"   - noc_embeddings.npy ({embeddings.nbytes / 1024 / 1024:.2f} MB)")
print(f"   - duty_embeddings.npy ({duty_embeddings.nbytes / 1024 / 1024:.2f} MB)")
print(f"   - noc_metadata.pkl")
