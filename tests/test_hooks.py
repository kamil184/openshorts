"""Tests for hook overlay text handling (emoji runs, long-word wrapping,
visual styles, bitmap-emoji scaling, image generation)."""
import os

from PIL import Image, ImageDraw, ImageFont

from hooks import (
    HOOK_STYLES,
    _break_long_word,
    _emoji_scale,
    _measure_width,
    _split_emoji_runs,
    _EMOJI_RE,
    create_hook_image,
)


def _draw_and_font():
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    return draw, ImageFont.load_default()


class TestEmojiRuns:
    def test_split_mixed_text(self):
        assert _split_emoji_runs("Feuer 🔥 test") == [
            (False, "Feuer "),
            (True, "🔥"),
            (False, " test"),
        ]

    def test_plain_text_single_run(self):
        assert _split_emoji_runs("nur text") == [(False, "nur text")]

    def test_emoji_only(self):
        assert _split_emoji_runs("🔥🚀") == [(True, "🔥🚀")]

    def test_strip_regex(self):
        assert _EMOJI_RE.sub("", "Stop 🛑 doing this! 💯") == "Stop  doing this! "


class TestLongWordWrap:
    def test_pieces_fit_and_recombine(self):
        draw, font = _draw_and_font()
        word = "A" * 60
        max_width = 50
        pieces = _break_long_word(draw, word, font, None, max_width)
        assert len(pieces) > 1
        assert "".join(pieces) == word
        for piece in pieces:
            assert draw.textlength(piece, font=font) <= max_width

    def test_short_word_single_piece(self):
        draw, font = _draw_and_font()
        assert _break_long_word(draw, "kurz", font, None, 1000) == ["kurz"]


class TestHookStyles:
    """The 6 styles are a contract shared with the frontend (HookModal's
    picker and HookOverlay's HOOK_LOOKS must mirror these keys)."""

    def test_expected_style_names(self):
        assert set(HOOK_STYLES) == {
            "headline", "pill", "classic", "dark", "yellow", "red", "outline", "outline_yellow",
        }

    def test_headline_is_boxless_condensed_with_accent_outline_and_shadow(self):
        look = HOOK_STYLES["headline"]
        assert look["box"][3] == 0
        assert look["shadow"] is True
        assert look["uppercase"] is True
        assert look["accent"] == (255, 229, 0)
        color, px = look["outline"]
        assert px > 0 and len(color) == 3
        assert look["font"].endswith("BarlowCondensed-ExtraBold.ttf")
        assert os.path.exists(look["font"])

    def test_pill_is_one_box_per_line_in_the_bundled_sans(self):
        look = HOOK_STYLES["pill"]
        assert look["pills"] is True
        assert look["font"].endswith("Montserrat-ExtraBold.ttf")
        assert os.path.exists(look["font"]), "the pill font must ship with the repo"

    def test_boxed_styles_have_opaque_box_and_shadow(self):
        for name in ("classic", "dark", "yellow", "red"):
            look = HOOK_STYLES[name]
            assert look["box"][3] > 0, name
            assert look["shadow"] is True, name
            assert look["outline"] is None, name

    def test_outline_styles_have_no_box_but_a_stroke(self):
        for name in ("outline", "outline_yellow"):
            look = HOOK_STYLES[name]
            assert look["box"][3] == 0, name
            assert look["shadow"] is False, name
            color, px = look["outline"]
            assert px > 0 and len(color) == 3, name


class TestEmojiScaling:
    """Fixed-bitmap emoji fonts (NotoColorEmoji only loads at its 109px
    strike) are rescaled to the text size instead of being dropped."""

    def _text_font(self, size=54):
        # A real scalable font so .size is meaningful.
        return ImageFont.truetype(os.path.join("fonts", "NotoSerif-Bold.ttf"), size)

    def test_scale_is_ratio_of_text_size_to_native_strike(self):
        font = self._text_font(54)
        assert _emoji_scale(font, (font, 109)) == 54 / 109

    def test_scalable_font_needs_no_rescale(self):
        font = self._text_font(54)
        assert _emoji_scale(font, (font, 54)) == 1.0

    def test_measure_width_scales_emoji_runs(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        font = self._text_font(54)
        emoji_font = self._text_font(109)  # stand-in for a 109px strike
        full = _measure_width(draw, "🔥", font, (emoji_font, 109))
        native = draw.textlength("🔥", font=emoji_font)
        assert abs(full - native * (54 / 109)) < 0.01

    def test_measure_width_without_emoji_font(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        font = self._text_font(54)
        assert _measure_width(draw, "hola", font, None) > 0


class TestCreateHookImage:
    def test_every_style_renders(self, tmp_path):
        for style in HOOK_STYLES:
            out = str(tmp_path / f"hook_{style}.png")
            path, w, h = create_hook_image(
                "Did you know? 🤯", 500, out, style=style)
            assert os.path.exists(path), style
            assert w > 0 and h > 0, style
            assert Image.open(path).size == (w, h), style

    def test_unknown_style_falls_back_to_classic(self, tmp_path):
        out = str(tmp_path / "hook_fallback.png")
        path, w, h = create_hook_image("Hola", 500, out, style="nope")
        assert os.path.exists(path) and w > 0 and h > 0


def test_pill_hook_image_stacks_one_box_per_line(tmp_path):
    """Two lines of text -> two rows of box pixels with a boxless column
    beside the shorter one (it is its own pill, not one shared card)."""
    from hooks import create_hook_image
    from PIL import Image
    out = tmp_path / "pill.png"
    path, w, h = create_hook_image("una frase bastante larga para partirse en dos lineas",
                                   600, str(out), style="pill")
    img = Image.open(path).convert("RGBA")
    assert (img.width, img.height) == (w, h)
    alpha_rows = [any(img.getpixel((x, y))[3] > 200 for x in range(0, img.width, 4))
                  for y in range(img.height)]
    assert sum(alpha_rows) > 0


def test_headline_parses_explicit_stars_and_strips_markers():
    from hooks import _parse_hook_paragraphs
    parsed = _parse_hook_paragraphs("He bought *four bows*?!", accent_auto=True, uppercase=True)
    assert parsed == [[
        ("HE", False),
        ("BOUGHT", False),
        ("FOUR", True),
        ("BOWS?!", True),
    ]]


def test_headline_auto_highlights_numbers_without_stars():
    from hooks import _parse_hook_paragraphs
    parsed = _parse_hook_paragraphs("He somehow bought four bows for free", accent_auto=True, uppercase=True)
    words = [w for p in parsed for w in p]
    accented = [w for w, is_acc in words if is_acc]
    assert accented == ["FOUR", "BOWS"]


def test_headline_auto_highlights_punch_word_without_numbers():
    from hooks import _parse_hook_paragraphs
    parsed = _parse_hook_paragraphs("Why everyone misses this secret", accent_auto=True, uppercase=True)
    words = [w for p in parsed for w in p]
    accented = [w for w, is_acc in words if is_acc]
    assert 1 <= len(accented) <= 2


def test_headline_renders_yellow_pixels_and_cyrillic_fallback(tmp_path):
    out_en = str(tmp_path / "headline_en.png")
    path_en, _, _ = create_hook_image("He bought four bows", 972, out_en, style="headline")
    im = Image.open(path_en).convert("RGBA")
    px = im.load()
    has_yellow = any(
        px[x, y][3] > 200 and px[x, y][0] > 220 and px[x, y][1] > 190 and px[x, y][2] < 80
        for y in range(0, im.height, 2) for x in range(0, im.width, 2)
    )
    assert has_yellow, "headline style must draw accented key words in yellow"

    # Cyrillic falls back to Montserrat ExtraBold without crashing or tofu.
    out_ru = str(tmp_path / "headline_ru.png")
    path_ru, w, h = create_hook_image("Он случайно купил *четыре лука*", 972, out_ru, style="headline")
    assert os.path.exists(path_ru) and w > 0 and h > 0


def test_auto_hook_defaults_to_headline_and_2_5s(monkeypatch, tmp_path):
    import inspect
    import hooks as hooks_mod
    import main as main_mod

    assert inspect.signature(hooks_mod.add_hook_to_video).parameters["style"].default == "headline"
    assert inspect.signature(hooks_mod.create_hook_image).parameters["style"].default == "headline"

    seen = {}

    def fake_add(video_path, text, output_path, **kw):
        seen.update(kw)
        with open(output_path, "wb") as f:
            f.write(b"v")
        return True

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"v")
    monkeypatch.setattr(hooks_mod, "add_hook_to_video", fake_add)
    monkeypatch.delenv("AUTO_HOOK_STYLE", raising=False)
    monkeypatch.delenv("AUTO_HOOK_SECONDS", raising=False)

    res = main_mod.auto_hook_clip(str(clip), {"viral_hook_text": "Watch this"})
    assert res is not None
    assert seen["style"] == "headline"
    assert seen["duration"] == 2.5
    assert res[1]["duration_seconds"] == 2.5
