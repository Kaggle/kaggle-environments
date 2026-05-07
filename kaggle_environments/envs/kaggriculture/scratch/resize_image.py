import sys
from PIL import Image

def resize_sprite(input_path, output_path):
    print(f"Resizing {input_path} -> {output_path}")
    img = Image.open(input_path)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.save(output_path, "PNG")
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resize_image.py <input_path> <output_path>")
        sys.exit(1)
    resize_sprite(sys.argv[1], sys.argv[2])
