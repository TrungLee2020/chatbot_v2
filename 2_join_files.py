import os
import shutil

# from datetime import date

# # Lấy ngày hiện tại
# date = date.today()

def copy_merge_folders(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_merge_folders(s, d)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def main():
    base_dir = "data_storage"
    source_folders = ["./gen_qa_from_splitted_doc", "./gen_summary", "./splitted_markdown_new", "./customer_qa"] 
    destination_folder = "./joined_results"

    for folder in source_folders:
        src_path = os.path.join(base_dir, folder)
        dst_path = os.path.join(base_dir, destination_folder)
        
        if os.path.exists(src_path):
            print(f"Copying from {folder} to {destination_folder}")
            copy_merge_folders(src_path, dst_path)
        else:
            print(f"Source folder {folder} does not exist. Skipping.")

    print("Merge complete.")



if __name__ == "__main__":
    main()