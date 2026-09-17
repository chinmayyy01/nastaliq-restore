"""
Pipeline: raw Urdu string -> PIL + raqm (HarfBuzz shaping + RTL bidi, live,
          driven by the font's own OpenType rules) -> rendered image
"""

from PIL import Image, ImageDraw, ImageFont
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = os.path.join(_REPO_ROOT, "fonts", "AwamiNastaliq-Regular.ttf")

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(
        f"Nastaliq font not found at {FONT_PATH}\n"
        "Download AwamiNastaliq-Regular.ttf (see README.md 'Font setup' "
        "section) and place it in the fonts/ folder at the repo root."
    )

SAMPLE_LINES = [
    "علم حاصل کرنا ہر مسلمان پر فرض ہے",       # "Seeking knowledge is obligatory upon every Muslim"
    "وقت کی قدر کرو یہ دوبارہ نہیں آتا",         # "Value time, it does not come again"
    "محنت میں عظمت ہے",                         # "There is greatness in hard work"
    "کتاب انسان کی بہترین دوست ہے",              # "The book is man's best friend"
    "صبر کا پھل میٹھا ہوتا ہے",                  # "The fruit of patience is sweet"
]


def render_line(text: str, font_size: int = 64, padding: int = 20) -> Image.Image:
    """Render a single line of raw Urdu text as a clean grayscale image.

    We pass the RAW text (no manual reshaping) and let raqm/HarfBuzz do the
    shaping via direction="rtl" + language="ur" -- this is what correctly
    triggers the font's ligature/joining rules.
    """
    font = ImageFont.truetype(FONT_PATH, font_size, layout_engine=ImageFont.Layout.RAQM)

    dummy_img = Image.new("L", (10, 10), color=255)
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font, direction="rtl", language="ur")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = Image.new("L", (text_w + 2 * padding, text_h + 2 * padding), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0,
               direction="rtl", language="ur")
    return img


def generate_dataset(out_dir: str = "data/clean", font_size: int = 64):
    """Render all sample lines and save them as clean ground-truth images."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, line in enumerate(SAMPLE_LINES):
        img = render_line(line, font_size=font_size)
        path = os.path.join(out_dir, f"line_{i:03d}.png")
        img.save(path)
        paths.append(path)
        print(f"Saved {path}  ({line})")
    return paths


if __name__ == "__main__":
    generate_dataset()