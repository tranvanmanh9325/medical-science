import os
from PIL import Image, ImageDraw, ImageFont

def generate_blender_gizmo_assets():
    os.makedirs('assets', exist_ok=True)
    ss = 1024
    size = 256
    
    try:
        font = ImageFont.truetype('arialbd.ttf', 440)
    except:
        font = ImageFont.load_default()

    # 1. Positive Pins (X, Y, Z)
    pos_pins = [
        ('pin_x.png', (234, 67, 53, 255), 'X'),    # Blender Coral Red
        ('pin_y.png', (52, 168, 83, 255), 'Y'),    # Blender Emerald Green
        ('pin_z.png', (66, 133, 244, 255), 'Z'),   # Blender Royal Blue
    ]

    for filename, color, letter in pos_pins:
        ss_img = Image.new('RGBA', (ss, ss), (0, 0, 0, 0))
        ss_draw = ImageDraw.Draw(ss_img)
        # Smooth circle
        ss_draw.ellipse([48, 48, ss-48, ss-48], fill=color)
        
        # Crisp White Letter
        bbox = ss_draw.textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (ss - tw) / 2 - bbox[0]
        ty = (ss - th) / 2 - bbox[1] - 8
        ss_draw.text((tx, ty), letter, fill=(255, 255, 255, 255), font=font)

        out = ss_img.resize((size, size), Image.Resampling.LANCZOS)
        out.save(os.path.join('assets', filename))
        print(f"Generated {filename}")

    # 2. Negative Pins (-X, -Y, -Z)
    neg_pins = [
        ('pin_neg_x.png', (197, 34, 31, 230)),
        ('pin_neg_y.png', (30, 142, 62, 230)),
        ('pin_neg_z.png', (26, 115, 232, 230)),
    ]
    for filename, color in neg_pins:
        ss_img = Image.new('RGBA', (ss, ss), (0, 0, 0, 0))
        ss_draw = ImageDraw.Draw(ss_img)
        ss_draw.ellipse([128, 128, ss-128, ss-128], fill=color)
        out = ss_img.resize((size, size), Image.Resampling.LANCZOS)
        out.save(os.path.join('assets', filename))
        print(f"Generated {filename}")

    # 3. Trackball Disc (Frosted Glass Background + Smooth Rim)
    ss_img = Image.new('RGBA', (ss, ss), (0, 0, 0, 0))
    ss_draw = ImageDraw.Draw(ss_img)
    # Fill
    ss_draw.ellipse([32, 32, ss-32, ss-32], fill=(30, 38, 48, 140))
    # Border
    ss_draw.ellipse([32, 32, ss-32, ss-32], outline=(150, 175, 205, 180), width=18)
    out = ss_img.resize((size, size), Image.Resampling.LANCZOS)
    out.save(os.path.join('assets', 'trackball_disc.png'))
    print("Generated trackball_disc.png")

    # 4. Center Pivot Dot
    ss_img = Image.new('RGBA', (ss, ss), (0, 0, 0, 0))
    ss_draw = ImageDraw.Draw(ss_img)
    ss_draw.ellipse([200, 200, ss-200, ss-200], fill=(255, 255, 255, 255))
    out = ss_img.resize((size, size), Image.Resampling.LANCZOS)
    out.save(os.path.join('assets', 'center_pivot.png'))
    print("Generated center_pivot.png")

if __name__ == '__main__':
    generate_blender_gizmo_assets()
