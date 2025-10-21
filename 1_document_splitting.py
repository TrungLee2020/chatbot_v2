
""" document splitting and metadata collection"""
import os
import glob
import pandas as pd
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import re
from typing import List, Tuple, Dict
from pathlib import Path
from dotenv import load_dotenv
import os
from check_file_duplicate import CheckFileDuplicate
from logger_config import CustomLogger
# Load environment variables
load_dotenv()

# Configure logging
logger = CustomLogger.get_logger(__name__)
tokenizer = AutoTokenizer.from_pretrained("Viet-Mistral/Vistral-7B-Chat")

# Define the folder mapping
new_folder_mapping = {
    "bd_vhx": "BƯU ĐIỆN VĂN HOÁ XÃ",
    "bccp_nd": "BƯU CHÍNH CHUYỂN PHÁT - NỘI ĐỊA",
    "bccp_qt": "BƯU CHÍNH CHUYỂN PHÁT - QUỐC TẾ",
    "hcc": "HÀNH CHÍNH CÔNG",
    "ppbl": "PHÂN PHỐI BÁN LẺ",
    "tcbc": "TÀI CHÍNH BƯU CHÍNH",
    "dtpt": "Dau_Tu_Phat_Trien",
    "ktcn": "Ky_Thuat_Cong_Nghe",
    "ncpt": "Nghien_Cuu_Phat_Trien_va_Thuong_Hieu",
    "qlcl": "Quan_Ly_Chat_Luong",
    "vptl": "Van_Phong_Tai_Lieu",
    "ttds": "Trung_Tam_Doi_Soat",
    "ktpc": "Kiem_Tra_Phap_Che",
    "temBC": "TemBC",
    "dvkh": "Dich_Vu_Khach_Hang",
    "tcns": "Tai_Chinh_Nhan_Su"
}

# Load metadata from all sheets
metadata_df_dict = pd.read_excel("data_storage/08may_merge/vnpost_doc_metadata_08may.xlsx", sheet_name=None)
OUTPUT_CSV_PATH = "data_storage/08may_merge/splitted_doc_metadata.csv"

# Base directories
base_input_dir = "data_storage/cleaned_markdown"
base_output_dir = "data_storage/splitted_markdown_new"

new_metadata = []
files_without_metadata = set()  # Change this to a set for efficient lookup

# Counter for total files and successfully processed files
total_files_found = 0
total_files_processed = 0


def check_existing_splitted(self, file_path: str, base_dir: str, new_base_dir: str) -> tuple[bool, str]:
    """Check if splitted document exists"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    splits = custom_split(content)
    try:
        input_path = Path(file_path)
        relative_path = os.path.relpath(file_path, base_dir)
        for i, split in enumerate(splits, start=1):
            if input_path.name.endswith('_part{i}'):
                return True, ""

            output_dir = os.path.join(new_base_dir, os.path.dirname(relative_path))
            base_name = os.path.basename(relative_path)
            summary_name = f"{base_name}_part{i}"
            output_path = os.path.join(output_dir, summary_name)

            if os.path.exists(output_path):
                return True, output_path

        return False, output_path
    except Exception as e:
        logger.error(f"Error checking existing summary: {e}")
        raise

def preprocess_markdown_with_placeholder(text: str, headers: List[Tuple[str, str]], placeholder: str = "UKN") -> str:
    lines = text.splitlines()
    processed_lines = []

    for i, line in enumerate(lines):
        processed_lines.append(line)
        
        # Check if the line is a header
        if any(line.startswith(symbol) for symbol, _ in headers):
            # If it's the last line or the next line is a header of the same or higher level
            if i + 1 >= len(lines) or any(
                lines[i + 1].startswith(symbol) and len(lines[i + 1].split()[0]) <= len(line.split()[0])
                for symbol, _ in headers
            ):
                # Add placeholder content
                processed_lines.append(placeholder)

    return "\n".join(processed_lines)

def clean_line_breaks(content):
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return '\n'.join(line for line in content.split('\n') if line.strip())

def is_table_header(line):
    return line.strip().startswith('|') and line.strip().endswith('|') and '-' in line

def split_large_table(table_content, max_rows=20):
    lines = table_content.strip().split('\n')
    header_end = next((i for i, line in enumerate(lines) if is_table_header(line)), -1)
    
    if header_end == -1:
        return [table_content]
    
    header = '\n'.join(lines[:header_end+1])
    body_lines = lines[header_end+1:]
    
    return [f"{header}\n" + '\n'.join(body_lines[i:i+max_rows]) 
            for i in range(0, len(body_lines), max_rows)]

def custom_split(content: str, window_size: int = 1) -> List[Dict]:
    """
    Split với context window - mỗi chunk sẽ có thêm context từ chunks xung quanh
    """
    content = clean_line_breaks(content)
    
    tokenizer = AutoTokenizer.from_pretrained("Viet-Mistral/Vistral-7B-Chat")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        length_function=lambda x: len(tokenizer.encode(x)),
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    headers_to_split_on = [(f"{'#' * i}", f"{'#' * i}") for i in range(1, 7)]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Preprocess the content with placeholders for empty headers
    preprocessed_content = preprocess_markdown_with_placeholder(content, headers_to_split_on)

    # Extract the document header (up to 3 lines before the first '#')
    lines = preprocessed_content.split('\n')
    doc_header_end = next((i for i, line in enumerate(lines) if line.startswith('#')), len(lines))
    doc_header = '\n'.join(lines[max(0, doc_header_end - 3):doc_header_end]).strip()
    
    # Create base chunks
    base_chunks = []
    try:
        markdown_splits = markdown_splitter.split_text(preprocessed_content)
        for split in markdown_splits:
            header = "\n".join([f"{k} {v}" for k, v in split.metadata.items() if v])
            combined_text = f"{header}\n\n{split.page_content.strip()}"
            
            if split.page_content.strip() != "UKN":
                text_splits = text_splitter.split_text(combined_text)
                base_chunks.extend([f"{doc_header}\n\n{s}" if not s.startswith(doc_header) else s 
                                   for s in text_splits])
    except:
        text_splits = text_splitter.split_text(preprocessed_content)
        base_chunks = [f"{doc_header}\n\n{s}" if not s.startswith(doc_header) else s 
                      for s in text_splits]
    
    # Clean chunks
    base_chunks = [s for s in base_chunks if s.strip()]
    base_chunks = [re.sub(r'^\n+', '', s) for s in base_chunks]
    base_chunks = [s.replace("\nUKN", "") for s in base_chunks]
    
    # Create chunks with context info
    chunks_with_context = []
    for i, chunk in enumerate(base_chunks):
        # Calculate context indices
        context_before_indices = list(range(max(0, i - window_size), i))
        context_after_indices = list(range(i + 1, min(len(base_chunks), i + 1 + window_size)))
        
        chunks_with_context.append({
            'text': chunk,
            'chunk_index': i,
            'total_chunks': len(base_chunks),
            'context_before_indices': context_before_indices,
            'context_after_indices': context_after_indices,
            # Lưu text của context luôn
            'context_before_text': [base_chunks[j] for j in context_before_indices],
            'context_after_text': [base_chunks[j] for j in context_after_indices]
        })
    
    return chunks_with_context
    # Remove any empty splits and ensure each split has at most 3 lines before the first header
    # splits = [s for s in splits if s.strip()]
    # splits = [re.sub(r'^\n+', '', s) for s in splits]
    # splits = ['\n'.join(s.split('\n')[:3] + [line for line in s.split('\n')[3:] if line.startswith('#') or not line.strip().startswith('Số:')]) for s in splits]
    
    # # Remove the placeholder text from the final splits
    # splits = [s.replace("\nUKN", "") for s in splits]
    
    # return splits

def process_file(file_path, output_dir, metadata):
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        logger.error(f"Encoding error with file: {file_path}. Trying with different encoding.")
        try:
            # Try with a different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {str(e)}")
            return []
    except Exception as e:
        logger.error(f"Failed to open file {file_path}: {str(e)}")
        return []
    
    chunks_data = custom_split(content, window_size=2)
    total_chunks = len(chunks_data)

    base_filename = os.path.basename(file_path)
    folder_path = os.path.relpath(os.path.dirname(file_path), base_input_dir)
    root_folder = folder_path.split(os.path.sep)[0] if folder_path != '.' else ''

    file_metadata = []
    for chunk_data in chunks_data:
        i = chunk_data['chunk_index']
        new_filename = f"{os.path.splitext(base_filename)[0]}_part{i+1}.txt"
        new_file_path = os.path.join(output_dir, new_filename)
        
        try:
            # Chỉ lưu main chunk vào file
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(chunk_data['text'])
        except Exception as e:
            logger.error(f"Failed to write file {new_file_path}: {str(e)}")
            continue
        
        # Lưu metadata với context info
        file_metadata.append({
            "file_type": metadata['file_type'],
            "filename": new_filename,
            "doc_id": metadata['doc_id'],
            "doc_title": metadata['doc_title'],
            "doc_date": str(metadata['doc_date']),
            "topic": metadata['topic'],
            "parent_content": chunk_data['text'],
            "folder": root_folder,
            "has_tables": metadata.get('has_tables', False),
            "tables_name": metadata.get('tables_name', 'N/A'),
            "tables_url": metadata.get('tables_url', 'N/A'),
            "is_used": metadata.get('is_used', True),
            "chunk_index": i,
            "total_chunks": total_chunks,
            "original_filename": base_filename,
            "context_before_indices": ','.join(map(str, chunk_data['context_before_indices'])),
            "context_after_indices": ','.join(map(str, chunk_data['context_after_indices'])),
            # Lưu text của context
            "context_before_text": "\n===CHUNK_SEP===\n".join(chunk_data['context_before_text']),
            "context_after_text": "\n===CHUNK_SEP===\n".join(chunk_data['context_after_text'])
        })
    
    return file_metadata

# main loop
new_metadata = []

# check duplicate file
file_processor = CheckFileDuplicate(base_input_dir, base_output_dir, OUTPUT_CSV_PATH)
for sheet_name, folder_name in new_folder_mapping.items():
    print(f"\nProcessing folder: {folder_name}")
    
    # Get the metadata for the current sheet
    metadata_df = metadata_df_dict[sheet_name]

    stats = file_processor.process_directory(folder_name, metadata_df)

    # Cập nhật tổng số
    total_files_found += stats['total']
    total_files_processed += stats['processed']
    # Check if 'tables_name' is present in this sheet
    has_tables_name = 'tables_name' in metadata_df.columns
    if not has_tables_name:
        print(f"Note: 'tables_name' column is missing in sheet '{sheet_name}'")
    
    # Get the input folder for the current sheet
    input_folder = os.path.join(base_input_dir, folder_name)
    
    # Get the metadata for the current sheet
    metadata_df = metadata_df_dict[sheet_name]
    
    folder_files_found = 0
    folder_files_processed = 0
    folder_files_without_metadata = set()  # Change to a set
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.txt'):
                folder_files_found += 1
                file_path = os.path.join(root, file)
                
                # Create corresponding output directory
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(base_output_dir, folder_name, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Find matching metadata
                base_filename = os.path.basename(file_path)
                matching_metadata = metadata_df[metadata_df['filename'] == base_filename]
                
                if matching_metadata.empty:
                    print(f"Warning: No metadata found for {file_path}. Creating default metadata.")
                    
                    # Tạo metadata tự động
                    parent_metadata = {
                        'file_type': 'auto_generated',
                        'filename': base_filename,
                        'doc_id': f'AUTO_{sheet_name}_{base_filename[:20]}',
                        'doc_title': base_filename.replace('.txt', ''),
                        'doc_date': 'N/A',
                        'topic': sheet_name,
                        'has_tables': False,
                        'tables_name': 'N/A',
                        'tables_url': 'N/A',
                        'is_used': True
                    }
                    
                    folder_files_without_metadata.add(file_path)
                    files_without_metadata.add(os.path.basename(file_path))
                    
                    # VẪN XỬ LÝ FILE NÀY thay vì skip
                    file_metadata = process_file(file_path, output_dir, parent_metadata)
                    if file_metadata:
                        new_metadata.extend(file_metadata)
                        folder_files_processed += 1
                else:
                    parent_metadata = matching_metadata.iloc[0].to_dict()
                    file_metadata = process_file(file_path, output_dir, parent_metadata)
                    if file_metadata:
                        new_metadata.extend(file_metadata)
                        folder_files_processed += 1

    
    print(f"Folder summary for {folder_name}:")
    print(f"  Files found: {folder_files_found}")
    print(f"  Files processed: {folder_files_processed}")
    print(f"  Files without metadata: {len(folder_files_without_metadata)}")
    
    if folder_files_without_metadata:
        print("  Files missing metadata in this folder:")
        for file_path in folder_files_without_metadata:
            print(f"    - {file_path}")
    
    total_files_found += folder_files_found
    files_without_metadata.update(folder_files_without_metadata)


print("\nOverall summary:")
print(f"Total files found: {total_files_found}")
print(f"Total files processed: {total_files_processed}")
print(f"Total files without metadata: {len(files_without_metadata)}")

# After the main loop:
for file_path in files_without_metadata:
    new_metadata.append({
        "file_type": "Unknown",
        "filename": os.path.basename(file_path),
        "doc_id": "Unknown",
        "doc_title": "Unknown",
        "doc_date": "Unknown",
        "topic": "Unknown",
        "parent_content": "Unknown",
        "folder": "Unknown",
        "has_tables": False,
        "tables_name": "N/A",
        "tables_url": "N/A",
        "is_used": False
    })


# Create a new metadata DataFrame
if new_metadata:
    new_metadata_df = pd.DataFrame(new_metadata)
    output_csv_path = OUTPUT_CSV_PATH
    new_metadata_df.to_csv(output_csv_path, index=False)
    print(f"\nProcessing complete. New metadata file saved to {output_csv_path}")
    print(f"New metadata entries created: {len(new_metadata)}")
    print(f"Files marked as not used: {len(files_without_metadata)}")
else:
    print("No metadata entries created.")


