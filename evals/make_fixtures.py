"""Build the two original, redistributable PDF exhibits."""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

ROOT = Path(__file__).parent / "fixtures"
REPORTS = [
    ("harbor", "Harbor pilot", "Trial site: Cedar Quay.\nInspection interval: 18 days.\nThe reference gauge is mounted on Pier K.", "Dock throughput", [("West dock", 42), ("East dock", 67), ("North dock", 29)], "crates"),
    ("orchard", "Orchard survey", "Survey leader: Mira Vale.\nThe calibration window lasts 36 hours.\nAll samples travel in amber cases.", "Plot yield", [("Birch plot", 34), ("Maple plot", 58), ("Alder plot", 21)], "kg"),
]


def build():
    ROOT.mkdir(exist_ok=True)
    for slug, title, facts, chart, rows, unit in REPORTS:
        doc = fitz.open()
        page = doc.new_page(width=540, height=480)
        page.insert_text((42, 68), title, fontsize=26, color=(0.12, .26, .34))
        page.insert_text((42, 125), facts, fontsize=15, lineheight=1.8)
        page = doc.new_page(width=540, height=480)
        page.insert_text((42, 66), f"Figure 1. {chart}", fontsize=23)
        im = Image.new("RGB", (900, 420), "#f3f6f5")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except OSError:
            font = ImageFont.load_default(size=30)
        for i, (label, value) in enumerate(rows):
            y = 55 + i * 110
            draw.text((25, y), label, font=font, fill="#193a42")
            draw.rectangle((220, y, 220 + value * 7, y + 48), fill="#127773")
            draw.text((235 + value * 7, y + 5), f"{value} {unit}", font=font, fill="#193a42")
        stream = BytesIO()
        im.save(stream, format="PNG")
        page.insert_image(fitz.Rect(32, 110, 508, 332), stream=stream.getvalue())
        doc.save(ROOT / f"{slug}.pdf", deflate=True, no_new_id=True)
        doc.close()


if __name__ == "__main__":
    build()
