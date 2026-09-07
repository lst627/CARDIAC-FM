from pathlib import Path

from PIL import Image, ImageOps, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "five_cmr_panels"
OUT.mkdir(exist_ok=True)


def font(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def gif_frame(path, idx):
    im = Image.open(path)
    im.seek(min(idx, getattr(im, "n_frames", idx + 1) - 1))
    return im.convert("L")


def crop_square(im, box=None, size=512):
    if box:
        im = im.crop(box)
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = ImageOps.autocontrast(im.convert("L"), cutoff=1)
    return im.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


fig2 = Image.open(ROOT / "jcmr_fig2_planes.jpg").convert("L")

panels = [
    ("2CH", crop_square(gif_frame(ROOT / "cardiac_2ch.gif", 10), size=512), "cmr_2ch_clean.png"),
    ("4CH", crop_square(gif_frame(ROOT / "cardiac_4ch.gif", 12), size=512), "cmr_4ch_clean.png"),
    ("SAX basal", crop_square(fig2, (730, 480, 903, 653), 512), "cmr_sax_basal_clean.png"),
    ("SAX mid", crop_square(fig2, (960, 374, 1121, 535), 512), "cmr_sax_mid_clean.png"),
    ("SAX apical", crop_square(fig2, (985, 122, 1150, 287), 512), "cmr_sax_apical_clean.png"),
]

for _, im, name in panels:
    im.save(OUT / name)

sheet = Image.new("RGB", (5 * 230 + 60, 300), "white")
d = ImageDraw.Draw(sheet)
d.text((24, 18), "Five representative CMR panels", font=font(24), fill=(15, 23, 42))
for i, (label, im, _) in enumerate(panels):
    thumb = im.resize((190, 190), Image.Resampling.LANCZOS)
    x = 30 + i * 230
    sheet.paste(thumb, (x, 66))
    d.rectangle((x, 66, x + 189, 255), outline=(15, 23, 42), width=2)
    d.text((x, 264), label, font=font(18), fill=(30, 41, 59))
sheet.save(OUT / "five_cmr_panels_contact_sheet.png")

(OUT / "sources.txt").write_text(
    "\n".join(
        [
            "2CH: Wikimedia Commons File:VLA.gif, CC BY-SA 4.0, https://commons.wikimedia.org/wiki/File:VLA.gif",
            "4CH: Wikimedia Commons File:4-CH cine normal.gif, source page https://commons.wikimedia.org/wiki/File:4-CH_cine_normal.gif",
            "SAX basal/mid/apical: Charoenpanichkit & Hundley, J Cardiovasc Magn Reson 12, 59 (2010), Figure 2, CC BY 2.0, https://doi.org/10.1186/1532-429X-12-59",
        ]
    )
    + "\n"
)

for _, _, name in panels:
    print(OUT / name)
print(OUT / "five_cmr_panels_contact_sheet.png")
