from PIL import Image
import os

def resize_images_in_folder(folder_path, size=(224, 224)):
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                try:
                    img = Image.open(img_path)
                    img = img.resize(size)
                    img.save(img_path)
                    print(f"Resized: {img_path}")
                except Exception as e:
                    print(f"Error resizing {img_path}: {e}")

resize_images_in_folder("dataset/fresh")
resize_images_in_folder("dataset/rotten")
