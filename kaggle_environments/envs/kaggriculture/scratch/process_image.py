import sys
from PIL import Image

def process_sprite(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    img = Image.open(input_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        # if it's close to white, make it transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.save(output_path, "PNG")
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_image.py <input_path> <output_path>")
        sys.exit(1)
    process_sprite(sys.argv[1], sys.argv[2])
