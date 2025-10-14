"""
    Code to Download all Train and Test images to the folder:
    'TrainImages' and 'TestImages' respectively
"""


import numpy as np
import pandas as pd
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial
import urllib.request
import time
import shutil
import socket

# ----------------------------
# Single Image Download
# ----------------------------
def download_image(row, savefolder, id_col, url_col, retries=3, timeout=10):
    """Download a single image with retry mechanism and timeout."""
    image_link = row[url_col]
    sample_id = str(row[id_col])
    filename = f"{sample_id}.jpg"
    image_save_path = os.path.join(savefolder, filename)

    # Skip if already exists and non-empty
    if os.path.exists(image_save_path) and os.path.getsize(image_save_path) > 0:
        return True

    for attempt in range(1, retries + 1):
        try:
            socket.setdefaulttimeout(timeout)  # Prevent hanging connections
            urllib.request.urlretrieve(image_link, image_save_path)
            return True
        except Exception as ex:
            if attempt < retries:
                time.sleep(1)
            else:
                with open(image_save_path, "wb") as f:
                    f.write(b"")
                print(f"⚠ Failed after {retries} attempts: {image_link} (ID: {sample_id}) - {ex}")
                return False


# ----------------------------
# Bulk Downloader with Safe Cleanup
# ----------------------------
def download_images(df, id_col, url_col, download_folder, retries=3, max_rounds=3):
    """Downloads all images with multiprocessing and ensures clean exit."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    all_records = df.to_dict("records")
    total_images = len(all_records)

    for round_no in range(1, max_rounds + 1):
        print(f"\n🚀 Download Round {round_no}/{max_rounds}")

        download_func = partial(
            download_image,
            savefolder=download_folder,
            id_col=id_col,
            url_col=url_col,
            retries=retries
        )

        results = []
        pool = multiprocessing.Pool(processes=16)  # Reduce workers to avoid hang
        try:
            for result in tqdm(pool.imap_unordered(download_func, all_records), total=len(all_records)):
                results.append(result)
        finally:
            pool.close()
            pool.join()
            pool.terminate()  # Force close if any stuck process remains

        success_count = sum(results)
        failed_count = total_images - success_count
        print(f"\n📊 Round {round_no} Summary: {success_count}/{total_images} succeeded, {failed_count} failed.")

        if failed_count == 0:
            print("\n✅ All images downloaded successfully!")
            break
        else:
            failed_rows = [r for i, r in enumerate(all_records) if not results[i]]
            all_records = failed_rows
            print(f"🔁 Retrying {len(failed_rows)} failed downloads in next round...")

    # ----------------------------
    # Fill Missing or Empty Images
    # ----------------------------
    print("\n🧩 Checking for missing images to fill by duplicating previous image...")

    all_ids = df[id_col].astype(str).tolist()
    last_valid_image = None
    duplicated_count = 0

    for sid in all_ids:
        img_path = os.path.join(download_folder, f"{sid}.jpg")

        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            if last_valid_image:
                shutil.copy(last_valid_image, img_path)
                duplicated_count += 1
            else:
                with open(img_path, "wb") as f:
                    f.write(b"")
                duplicated_count += 1
        else:
            last_valid_image = img_path

    # ----------------------------
    # Final Summary
    # ----------------------------
    final_files = [f for f in os.listdir(download_folder) if os.path.getsize(os.path.join(download_folder, f)) > 0]
    print("\n📦 Final Download Report:")
    print(f"  ➤ Total Expected: {total_images}")
    print(f"  ✅ Downloaded:     {len(final_files)}")
    print(f"  🔁 Duplicated:     {duplicated_count}")
    print(f"  📁 Images saved in: {download_folder}")

    return len(final_files), duplicated_count


# ----------------------------
# Main Execution for Kaggle
# ----------------------------
def main():
    train_path = "dataset/train.csv" # "dataset/test.csv" for test images
    train_df = pd.read_csv(train_path)

    id_col = "sample_id"
    url_col = "image_link"

    print(f"📁 Total images to download: {len(train_df)}")

    download_folder = "./TrainImages" # "./TestImages" for test images
    
    for chunk in np.array_split(train_df, 5):
        download_images(chunk, id_col, url_col, download_folder)



if __name__ == "__main__":
    main()