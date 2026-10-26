"""Generate deterministic demo screenshots for the web UI and examples folder.

Run from the project root: python scripts/make_demo_assets.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "client" / "web" / "assets"
EXAMPLES = ROOT / "examples"
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT
    if not Path(path).exists():
        path = "C:/Windows/Fonts/simhei.ttf"
    return ImageFont.truetype(path, size)


def radial_glow(size: tuple[int, int], center: tuple[int, int], radius: int, color: tuple[int, int, int], strength: float) -> Image.Image:
    width, height = size
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    alpha = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 2 * strength
    layer = np.zeros((height, width, 3), dtype=np.uint8)
    layer[:, :] = color
    image = Image.fromarray(layer, "RGB").convert("RGBA")
    image.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    return image


def make_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = 8 + 12 * y + 3 * x
    base[:, :, 1] = 13 + 18 * y + 6 * x
    base[:, :, 2] = 21 + 28 * y + 8 * x
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    image.alpha_composite(radial_glow(size, (1120, 330), 560, (42, 165, 155), 0.28))
    image.alpha_composite(radial_glow(size, (240, 760), 500, (160, 69, 74), 0.12))
    image.alpha_composite(radial_glow(size, (1230, 790), 340, (194, 146, 74), 0.08))
    return image


def add_stars(image: Image.Image, count: int = 120) -> None:
    rng = random.Random(41)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for _ in range(count):
        x = rng.randint(0, width)
        y = rng.randint(0, int(height * 0.72))
        radius = rng.choice([0.6, 0.8, 1.0, 1.4])
        alpha = rng.randint(28, 115)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(196, 238, 232, alpha))


def add_portal(image: Image.Image, center: tuple[int, int]) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    cx, cy = center
    # Outer atmospheric haze.
    for radius, alpha in [(320, 14), (250, 22), (195, 28)]:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(53, 217, 200, alpha))
    # Broken rings and runes.
    for radius, color, width in [(210, (78, 231, 214, 100), 2), (176, (222, 243, 204, 65), 1), (132, (78, 231, 214, 120), 2)]:
        start = -90
        for index in range(6):
            extent = 44 if index % 2 == 0 else 19
            draw.arc(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                start,
                start + extent,
                fill=color,
                width=width,
            )
            start += 60
    # Central light.
    for radius, alpha in [(108, 50), (78, 95), (46, 180), (20, 245)]:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(202, 255, 236, alpha))
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x1 = cx + math.cos(rad) * 232
        y1 = cy + math.sin(rad) * 232
        x2 = cx + math.cos(rad) * 258
        y2 = cy + math.sin(rad) * 258
        draw.line((x1, y1, x2, y2), fill=(152, 244, 225, 80), width=2)

    glow = overlay.filter(ImageFilter.GaussianBlur(18))
    image.alpha_composite(glow)
    image.alpha_composite(overlay)


def add_figure(image: Image.Image) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    # A cloaked traveller silhouette, deliberately graphic rather than detailed.
    draw.ellipse((1006, 254, 1102, 350), fill=(12, 16, 22, 235))
    draw.polygon(
        [(1054, 330), (1126, 826), (954, 826), (1006, 356)],
        fill=(10, 14, 20, 238),
    )
    draw.polygon(
        [(1054, 344), (1088, 548), (1118, 818), (1040, 818), (1028, 560), (1006, 380)],
        fill=(22, 32, 40, 170),
    )
    draw.line((1018, 494, 968, 672), fill=(12, 16, 22, 235), width=30)
    draw.line((1091, 497, 1148, 654), fill=(12, 16, 22, 235), width=28)
    # Hair strands.
    for offset in range(-3, 4):
        draw.arc(
            (1006 + offset * 2, 246, 1104 + offset * 2, 410),
            200,
            330,
            fill=(86, 110, 114, 120),
            width=2,
        )
    image.alpha_composite(overlay)


def rounded_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int] = (12, 17, 24, 205),
    outline: tuple[int, int, int, int] = (130, 210, 200, 32),
    radius: int = 22,
    width: int = 1,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    image.alpha_composite(overlay)


def add_quest_panel(image: Image.Image) -> None:
    rounded_panel(image, (58, 86, 370, 292), fill=(11, 16, 22, 198))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((84, 112, 92, 252), radius=4, fill=(224, 169, 88, 220))
    draw.text((112, 108), "主线任务", font=font(18, True), fill=(228, 177, 100, 255))
    draw.text((110, 148), "星辉遗迹", font=font(36, True), fill=(242, 240, 231, 255))
    draw.text((111, 202), "寻找失落的星图", font=font(20), fill=(177, 191, 193, 255))
    draw.text((111, 240), "封印之门 · 未完成", font=font(14), fill=(112, 137, 141, 255))
    draw.ellipse((306, 120, 338, 152), outline=(224, 169, 88, 100), width=1)
    draw.polygon([(322, 128), (327, 140), (322, 147), (317, 140)], fill=(242, 199, 124, 180))


def add_party_panel(image: Image.Image) -> None:
    rounded_panel(image, (58, 736, 1342, 840), fill=(9, 13, 18, 224), outline=(194, 161, 96, 40))
    draw = ImageDraw.Draw(image, "RGBA")
    # Speaker badge.
    draw.rounded_rectangle((88, 758, 234, 806), radius=12, fill=(26, 37, 44, 255), outline=(92, 211, 194, 90))
    draw.text((108, 766), "艾莉娅", font=font(22, True), fill=(134, 238, 218, 255))
    draw.text((276, 762), "旅人，你终于来了。", font=font(30, True), fill=(242, 239, 228, 255))
    draw.text(
        (276, 802),
        "前面的路被封印了，只有你能解开它。",
        font=font(21),
        fill=(173, 187, 187, 255),
    )
    # Keyboard hint.
    draw.rounded_rectangle((1172, 780, 1317, 818), radius=8, fill=(255, 255, 255, 10), outline=(255, 255, 255, 34))
    draw.text((1190, 788), "空格", font=font(13), fill=(206, 208, 201, 230))
    draw.text((1242, 789), "继续", font=font(13), fill=(132, 148, 149, 230))


def add_top_hud(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((58, 40), "AETHERIA", font=font(14, True), fill=(120, 223, 206, 190))
    draw.text((160, 41), "/  CHAPTER IV", font=font(11), fill=(99, 112, 116, 180))
    draw.text((1148, 41), "23:48", font=font(12), fill=(106, 122, 125, 180))
    draw.ellipse((1128, 43, 1138, 53), outline=(224, 169, 88, 140), width=1)
    # Bottom-left ability icons.
    for index, label in enumerate(["Q", "E", "R"]):
        x = 1168 + index * 52
        y = 690
        draw.ellipse((x, y, x + 38, y + 38), fill=(11, 16, 22, 175), outline=(255, 255, 255, 35))
        draw.text((x + 14, y + 9), label, font=font(13, True), fill=(152, 166, 166, 210))
    # Decorative fine lines.
    draw.line((58, 318, 370, 318), fill=(255, 255, 255, 16), width=1)
    draw.line((58, 708, 1342, 708), fill=(194, 161, 96, 28), width=1)


def make_game_demo() -> Image.Image:
    size = (1400, 900)
    image = make_background(size)
    add_stars(image)
    add_portal(image, (1110, 360))
    add_figure(image)
    # Haze in front of the figure.
    haze = radial_glow(size, (1110, 420), 430, (36, 108, 104), 0.18)
    image.alpha_composite(haze)
    add_quest_panel(image)
    add_party_panel(image)
    add_top_hud(image)
    # Vignette.
    vignette = Image.new("L", size, 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse((-260, -240, 1660, 1160), fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(160))
    black = Image.new("RGBA", size, (0, 0, 0, 155))
    black.putalpha(Image.eval(vignette, lambda value: 155 - int(value * 0.42)))
    image.alpha_composite(black)
    return image.convert("RGB")


def make_manga_demo() -> Image.Image:
    size = (900, 1280)
    image = Image.new("RGB", size, (238, 234, 224))
    draw = ImageDraw.Draw(image, "RGBA")
    # Panel borders.
    margin = 30
    gutter = 16
    top_h = 650
    draw.rectangle((margin, margin, size[0] - margin, margin + top_h), fill=(245, 242, 234), outline=(25, 25, 25), width=4)
    draw.rectangle((margin, margin + top_h + gutter, size[0] - margin, size[1] - margin), fill=(241, 237, 228), outline=(25, 25, 25), width=4)
    # Screen tone using dot pattern in the top panel.
    tone = Image.new("RGBA", (size[0] - 2 * margin, top_h), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(tone, "RGBA")
    for y in range(0, top_h, 9):
        for x in range(0, size[0] - 2 * margin, 9):
            tdraw.ellipse((x, y, x + 2, y + 2), fill=(29, 32, 38, 55))
    tone = tone.rotate(0)
    image.paste(tone, (margin, margin), tone)
    # Moon and skyline.
    draw.ellipse((530, 90, 760, 320), fill=(250, 247, 235, 255), outline=(25, 25, 25, 255), width=4)
    draw.polygon(
        [(30, 660), (30, 500), (120, 430), (195, 520), (270, 390), (380, 520), (470, 460), (560, 540), (650, 480), (740, 560), (870, 500), (870, 660)],
        fill=(26, 30, 36, 255),
    )
    # Speed lines.
    for index in range(26):
        y = 70 + index * 19
        draw.line((70, y, 380 + (index % 5) * 30, y - 20), fill=(30, 30, 30, 28), width=1)
    # Vertical speech bubble top-right.
    draw.ellipse((604, 344, 830, 620), fill=(255, 253, 246, 245), outline=(25, 25, 25), width=4)
    draw.polygon([(630, 590), (585, 646), (669, 612)], fill=(255, 253, 246, 245), outline=(25, 25, 25))
    # Vertical Japanese/Chinese dialogue (one char per line, right-to-left).
    vertical = "旅人よ、星の扉が開く"
    x = 788
    for column_start in range(0, len(vertical), 10):
        column = vertical[column_start : column_start + 10]
        for index, char in enumerate(column):
            draw.text((x, 376 + index * 24), char, font=font(23, True), fill=(24, 24, 24, 255))
        x -= 29
    # Bottom panel: close-up eye illustration, abstract.
    draw.ellipse((140, 810, 740, 1110), fill=(225, 219, 205, 255), outline=(25, 25, 25), width=4)
    draw.ellipse((260, 864, 620, 1058), fill=(250, 247, 238, 255), outline=(25, 25, 25), width=4)
    draw.ellipse((370, 900, 508, 1022), fill=(32, 34, 37, 255))
    draw.ellipse((410, 932, 472, 992), fill=(204, 163, 91, 255))
    draw.ellipse((427, 948, 458, 979), fill=(15, 15, 17, 255))
    draw.ellipse((442, 944, 452, 954), fill=(255, 248, 220, 230))
    for index in range(9):
        draw.arc((235 - index * 2, 840 - index * 2, 650 + index * 2, 1090 + index * 2), 190, 350, fill=(60, 61, 60, 80), width=1)
    # Caption box.
    draw.rectangle((60, 1132, 840, 1214), fill=(17, 18, 21, 255))
    draw.text((84, 1150), "那一天，我们看见了星海的尽头。", font=font(27, True), fill=(245, 241, 229, 255))
    # Page number.
    draw.text((792, 1232), "07", font=font(18, True), fill=(89, 87, 81, 255))
    return image


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    game = make_game_demo()
    game.save(ASSETS / "demo.png", optimize=True)
    game.save(EXAMPLES / "demo_game_ui.png", optimize=True)
    manga = make_manga_demo()
    manga.save(EXAMPLES / "demo_manga.png", optimize=True)
    print("Wrote", ASSETS / "demo.png")
    print("Wrote", EXAMPLES / "demo_game_ui.png")
    print("Wrote", EXAMPLES / "demo_manga.png")


if __name__ == "__main__":
    main()
