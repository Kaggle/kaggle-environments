import sys
from PIL import Image

def crop_and_scale(input_path, output_path):
    print(f"Cropping and scaling {input_path} -> {output_path}")
    img = Image.open(input_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    # First, make white transparent
    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)

    # Get bounding box of non-transparent part
    bbox = img.getbbox()
    if not bbox:
        print("Error: Image is empty or all white.")
        return
    
    print(f"Bounding box: {bbox}")
    cropped_img = img.crop(bbox)
    
    # Calculate new size to fit in 128x128 while maintaining aspect ratio
    width, height = cropped_img.size
    max_dim = max(width, height)
    scale_factor = 120 / max_dim # leave a small margin
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_cropped = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new 128x128 transparent image and paste resized cow in center
    final_img = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
    paste_x = (128 - new_width) // 2
    paste_y = (128 - new_height) // 2
    final_img.paste(resized_cropped, (paste_x, paste_y), resized_cropped)
    
    final_img.save(output_path, "PNG")
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python crop_and_scale.py <input_path> <output_path>")
        sys.exit(1)
    crop_and_scale(sys.argv[1], sys.argv[2])
