import os
import pandas as pd
import csv
import chardet
import re

# Define the paths
BASE_DATA_DIR = "data_storage"
BASE_METADATA_DIR = os.path.join("data_storage")
METADATA_DIR = os.path.join(BASE_METADATA_DIR, "08may_merge")

joined_results_path = os.path.join(BASE_DATA_DIR, "joined_results")
docs_path = BASE_DATA_DIR
original_metadata_path = os.path.join(METADATA_DIR, "vnpost_doc_metadata_08may.xlsx")
splitted_metadata_path = os.path.join(METADATA_DIR, "splitted_doc_meta_data_1210.csv")
qa_from_splitted_doc_metadata_path = os.path.join(METADATA_DIR, "qa_from_splitted_doc_meta_data_1210.csv")
COMBINED_METADATA_PATH = os.path.join(METADATA_DIR, "combined_metadata_1210.csv")


# Define the folder mapping
folder_mapping = {
    "bd_vhx": "BƯU ĐIỆN VĂN HOÁ XÃ",
    "bccp_nd": "BƯU CHÍNH CHUYỂN PHÁT - NỘI ĐỊA",
    "bccp_qt": "BƯU CHÍNH CHUYỂN PHÁT - QUỐC TẾ",
    "tcbc": "TÀI CHÍNH BƯU CHÍNH",
    "ppbl": "PHÂN PHỐI BÁN LẺ",
    "hcc": "HÀNH CHÍNH CÔNG",
    "dtpt": "Dau_Tu_Phat_Trien",
    "ncpt": "Nghien_Cuu_Phat_Trien_va_Thuong_Hieu",
    "qlcl": "Quan_Ly_Chat_Luong",
    "ktcn": "Ky_Thuat_Cong_Nghe",
    "vptl": "Van_Phong_Tai_Lieu",
    "ttds": "Trung_Tam_Doi_Soat",
    "temBC": "TemBC",
    "ktpc": "Kiem_Tra_Phap_Che",
    "dvkh": "Dich_Vu_Khach_Hang",
    "tcns": "Tai_Chinh_Nhan_Su"
}

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def sanitize_filename(filename):
    if not isinstance(filename, str):
        return filename
    
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    sanitized = sanitized.strip('. ')
    
    return sanitized

def generate_metadata_for_qa_summary(file_path, original_df):
    base_name = os.path.basename(file_path)
    file_type = 'question_answering' if base_name.startswith('QA-') else 'summary'
    
    original_doc_name = re.sub(r'^(QA-|SUMMARY-)', '', base_name)
    original_doc_name = original_doc_name.replace('.txt', '')
    
    # Use exact matching instead of contains
    matching_metadata = original_df[original_df['filename'].str.lower() == original_doc_name.lower()]
    
    if not matching_metadata.empty:
        metadata = matching_metadata.iloc[0].to_dict()
        metadata['file_type'] = file_type
        metadata['filename'] = base_name
        return metadata
    else:
        return {
            'file_type': file_type,
            'filename': base_name,
            'doc_id': 'N/A',
            'doc_title': original_doc_name,
            'doc_date': 'N/A',
            'topic': 'N/A',
            'tables_name': 'N/A',
            'tables_url': 'N/A'
        }


def safe_read_csv_robust(file_path):
    """Đọc CSV với xử lý dấu phẩy và quote đúng cách"""
    try:
        # Detect encoding
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']

        rows = []
        skipped = 0
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            # Dùng csv.reader để parse đúng quote
            reader = csv.reader(f, quotechar='"', delimiter=',', 
                              doublequote=True, skipinitialspace=True)
            
            # Read header
            header = next(reader)
            num_columns = len(header)
            print(f"Reading {os.path.basename(file_path)}: {num_columns} columns expected")
            
            # Read data rows
            for line_num, row in enumerate(reader, start=2):
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                # Fix row length
                if len(row) < num_columns:
                    # Thiếu cột -> thêm empty string
                    row.extend([''] * (num_columns - len(row)))
                elif len(row) > num_columns:
                    # Thừa cột -> merge vào cột cuối
                    if len(row) <= num_columns + 5:  # Chỉ fix nếu thừa <= 5 cột
                        row = row[:num_columns-1] + [','.join(row[num_columns-1:])]
                    else:
                        # Thừa quá nhiều -> skip
                        skipped += 1
                        if skipped <= 3:
                            print(f"  ⚠️ Skipped line {line_num}: {len(row)} cols (expected {num_columns})")
                        continue
                
                rows.append(row)
        
        df = pd.DataFrame(rows, columns=header)
        print(f"  ✓ Loaded: {len(df)} rows | Skipped: {skipped}")
        return df

    except Exception as e:
        print(f"Failed to read {os.path.basename(file_path)}. Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Load original metadata
try:
    original_metadata = pd.read_excel(original_metadata_path, sheet_name=None)
    original_df = pd.concat(original_metadata.values(), ignore_index=True)
    original_df = original_df[original_df['file_type'] != 'original_markdown']
except Exception as e:
    print(f"Error loading original metadata: {e}")
    original_df = pd.DataFrame()

# Load splitted metadata
try:
    splitted_df = safe_read_csv_robust(splitted_metadata_path)  # ← SỬA TÊN HÀM
    print(f"✓ Loaded splitted metadata: {splitted_df.shape}")
except Exception as e:
    print(f"Error loading splitted metadata: {e}")
    import traceback
    traceback.print_exc()
    splitted_df = pd.DataFrame()

# Load QA from chunks metadata
try:
    qa_df = safe_read_csv_robust(qa_from_splitted_doc_metadata_path)  # ← SỬA TÊN HÀM
    print(f"✓ Loaded QA metadata: {qa_df.shape}")
except Exception as e:
    print(f"Error loading QA metadata: {e}")
    import traceback
    traceback.print_exc()
    qa_df = pd.DataFrame()

# Ensure all dataframes have the same columns
columns_to_use = ['file_type', 'filename', 'doc_id', 'doc_title', 'doc_date', 'topic', 'tables_name', 'tables_url', 'parent_content']
original_df = original_df.reindex(columns=columns_to_use, fill_value='N/A')
splitted_df = splitted_df.reindex(columns=columns_to_use)
qa_df = qa_df.reindex(columns=columns_to_use)

# Combine metadata
combined_df = pd.concat([original_df, splitted_df, qa_df], ignore_index=True)

# Remove rows with is_used == False (if this column exists)
if 'is_used' in combined_df.columns:
    combined_df = combined_df[combined_df['is_used'] != False]

# Sanitize filenames
combined_df['filename'] = combined_df['filename'].apply(sanitize_filename)

all_files = []
files_without_metadata = []

for sheet_name, folder_name in folder_mapping.items():
    print(f"\nProcessing folder: {folder_name}")
    
    input_folder = os.path.join(joined_results_path, folder_name)
    
    if not os.path.exists(input_folder):
        print(f"Folder does not exist: {input_folder}")
        continue
    
    folder_files_found = 0
    folder_files_processed = 0
    folder_files_without_metadata = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            folder_files_found += 1
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, joined_results_path)
            all_files.append(relative_path)
            
            if os.path.basename(file) not in set(combined_df['filename']):
                if file.startswith('QA-') or file.startswith('SUMMARY-'):
                    new_metadata = generate_metadata_for_qa_summary(file_path, original_df)
                    combined_df = pd.concat([combined_df, pd.DataFrame([new_metadata])], ignore_index=True)
                    folder_files_processed += 1
                else:
                    folder_files_without_metadata.append(file_path)
                    files_without_metadata.append(relative_path)
            else:
                folder_files_processed += 1
    
    print(f"Folder summary for {folder_name}:")
    print(f"  Files found: {folder_files_found}")
    print(f"  Files processed: {folder_files_processed}")
    print(f"  Files without metadata: {len(folder_files_without_metadata)}")

combined_df.drop_duplicates(subset='filename', keep='first', inplace=True)

print("\nOverall summary:")
print(f"Total files found: {len(all_files)}")
print(f"Total files processed: {len(all_files) - len(files_without_metadata)}")
print(f"Total files without metadata: {len(files_without_metadata)}")

# Print detailed info for files without metadata
print("\nDetailed info for files without metadata:")
for file in files_without_metadata:
    full_path = os.path.join(joined_results_path, file)
    file_size = os.path.getsize(full_path)
    file_mtime = os.path.getmtime(full_path)
    print(f"File: {file}")
    print(f"  Full path: {full_path}")
    print(f"  Size: {file_size} bytes")
    print(f"  Last modified: {pd.Timestamp(file_mtime, unit='s')}")
    print(f"  Content preview:")
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read(200))  # Print first 200 characters
    except Exception as e:
        print(f"  Error reading file: {e}")
    print()

print("\nUnique prefixes of files without metadata:")
prefixes = set(os.path.splitext(os.path.basename(f))[0].split('-')[0] for f in files_without_metadata)
print(', '.join(prefixes))

combined_df['parent_content'] = combined_df['parent_content'].fillna('N/A')

# Save the combined DataFrame to a CSV file
try:
    combined_df.to_csv(COMBINED_METADATA_PATH, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"\nCombined metadata saved to {COMBINED_METADATA_PATH}")
    print(f"Total rows in combined metadata: {len(combined_df)}")
except Exception as e:
    print(f"Error saving combined metadata: {e}")

print("\nColumns in the combined metadata:")
print(combined_df.columns.tolist())
print("\nSample of the combined metadata:")
print(combined_df.head().to_string())
