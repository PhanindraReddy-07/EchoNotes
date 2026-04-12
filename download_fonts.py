"""
Run this once to download multilingual fonts for EchoNotes PDF support.
Place this file at: C:\\Users\\PHANINDRA\\Desktop\\final year project\\
Then run: python download_fonts.py
"""
import urllib.request
import os
from pathlib import Path

# Create fonts/ folder next to this script
FONTS_DIR = Path(__file__).parent / "fonts"
FONTS_DIR.mkdir(exist_ok=True)

FONTS = {
    # Covers Telugu + Latin (primary fix)
    "NotoSansTelugu-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf",
    "NotoSansTelugu-Bold.ttf":    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Bold.ttf",
    # Covers Hindi/Devanagari + Latin
    "NotoSansDevanagari-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoSansDevanagari-Bold.ttf":    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
}

print("Downloading fonts for EchoNotes multilingual PDF support...\n")

for filename, url in FONTS.items():
    dest = FONTS_DIR / filename
    if dest.exists():
        print(f"  ✅ Already exists: {filename}")
        continue
    print(f"  ⬇️  Downloading {filename}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb} KB)")
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nFonts saved to: {FONTS_DIR}")
print("Telugu and Hindi PDFs will now render correctly.")
