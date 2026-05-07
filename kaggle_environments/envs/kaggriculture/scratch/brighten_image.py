import sys
from PIL import Image, ImageEnhance

def brighten_sprite(input_path, output_path, factor):
    print(f"Brightening {input_path} by {factor} -> {output_path}")
    img = Image.open(input_path)
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(factor)
    img_bright = img_bright.resize((128, 128), Image.Resampling.LANCZOS)
    img_bright.save(output_path, "PNG")
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python brighten_image.py <input_path> <output_path> <factor>")
        sys.exit(1)
    brighten_sprite(sys.argv[1], sys.argv[2], float(sys.argv[3]))
