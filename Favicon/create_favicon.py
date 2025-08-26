#!/usr/bin/env python3
"""
Create a favicon.ico file with black to orange gradient and white JT text
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_favicon():
    # Create a 32x32 image
    size = 32
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create gradient effect (simplified)
    for y in range(size):
        for x in range(size):
            # Calculate gradient from black to orange
            ratio = (x + y) / (size * 2)
            r = int(255 * ratio)  # Orange component
            g = int(102 * ratio)  # Orange component
            b = 0
            draw.point((x, y), fill=(r, g, b, 255))
    
    # Add white JT text
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Calculate text position to center it
    text = "JT"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # Draw white text with black outline for better visibility
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    # Save as favicon.ico
    img.save('favicon.ico', format='ICO', sizes=[(16, 16), (32, 32)])
    print("✅ Favicon created successfully!")
    print("📁 File: favicon.ico")
    print("🔄 Refresh your website to see it!")

if __name__ == "__main__":
    try:
        create_favicon()
    except ImportError:
        print("❌ PIL/Pillow not installed. Installing...")
        os.system("pip3 install Pillow")
        print("🔄 Please run the script again after installation.")
    except Exception as e:
        print(f"❌ Error: {e}")
