import sys
from PIL import Image

def combine_sprites(base_path, icon_path, output_path, x, y, icon_size):
    print(f"Combining {base_path} + {icon_path} -> {output_path}")
    base_img = Image.open(base_path).convert("RGBA")
    icon_img = Image.open(icon_path).convert("RGBA")
    
    # Make white transparent on the icon
    datas = icon_img.getdata()
    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    icon_img.putdata(newData)
    
    # Get bounding box to crop empty space
    bbox = icon_img.getbbox()
    if bbox:
        icon_img = icon_img.crop(bbox)
    
    # Resize icon to fit
    icon_img = icon_img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
    
    # Paste on base
    base_img.paste(icon_img, (x, y), icon_img)
    base_img.save(output_path, "PNG")
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python combine_sprites.py <base_path> <icon_path> <output_path> <x> <y> <icon_size>")
        sys.exit(1)
    combine_sprites(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
