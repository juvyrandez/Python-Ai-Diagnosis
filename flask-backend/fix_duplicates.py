import pandas as pd

def clean_duplicate_translations(text):
    """Remove duplicate Cebuano translations"""
    # Split by ' / ' separator
    parts = text.split(' / ')
    
    # Keep only first two parts (English and first Cebuano)
    if len(parts) >= 2:
        return f"{parts[0]} / {parts[1]}"
    else:
        return text

# Read and fix complaints_expanded.csv
print("Fixing duplicate translations in complaints_expanded.csv...")
df = pd.read_csv('complaints_expanded.csv')

# Clean complaint column
df['complaint'] = df['complaint'].apply(clean_duplicate_translations)

# Save
df.to_csv('complaints_expanded.csv', index=False)

print(f"✓ Fixed {len(df)} rows")
print(f"Sample: {df['complaint'].iloc[0]}")
