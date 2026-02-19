import pytest
from prot.hass import parse_color


class TestParseColor:
    def test_named_english_red(self):
        assert parse_color("red") == [255, 0, 0]

    def test_named_english_blue(self):
        assert parse_color("blue") == [0, 0, 255]

    def test_named_english_warm(self):
        assert parse_color("warm") == [255, 180, 107]

    def test_named_english_cool(self):
        assert parse_color("cool") == [166, 209, 255]

    def test_named_korean_빨강(self):
        assert parse_color("빨강") == [255, 0, 0]

    def test_named_korean_파랑(self):
        assert parse_color("파랑") == [0, 0, 255]

    def test_named_korean_초록(self):
        assert parse_color("초록") == [0, 128, 0]

    def test_named_korean_노랑(self):
        assert parse_color("노랑") == [255, 255, 0]

    def test_named_korean_분홍(self):
        assert parse_color("분홍") == [255, 192, 203]

    def test_named_korean_보라(self):
        assert parse_color("보라") == [128, 0, 128]

    def test_named_korean_주황(self):
        assert parse_color("주황") == [255, 165, 0]

    def test_named_korean_하양(self):
        assert parse_color("하양") == [255, 255, 255]

    def test_named_korean_흰색(self):
        assert parse_color("흰색") == [255, 255, 255]

    def test_hex_with_hash(self):
        assert parse_color("#FF0000") == [255, 0, 0]

    def test_hex_without_hash(self):
        assert parse_color("00FF00") == [0, 255, 0]

    def test_hex_lowercase(self):
        assert parse_color("#ff8800") == [255, 136, 0]

    def test_rgb_format(self):
        assert parse_color("rgb(255, 0, 0)") == [255, 0, 0]

    def test_hsl_format(self):
        result = parse_color("hsl(120, 100, 50)")
        assert result == [0, 255, 0]

    def test_hsl_bare_format(self):
        result = parse_color("120, 100, 50")
        assert result == [0, 255, 0]

    def test_unknown_returns_none(self):
        assert parse_color("xyzzy") is None

    def test_empty_returns_none(self):
        assert parse_color("") is None

    def test_case_insensitive(self):
        assert parse_color("RED") == [255, 0, 0]
        assert parse_color("Blue") == [0, 0, 255]
