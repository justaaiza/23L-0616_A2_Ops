import io

import pytest
from PIL import Image

from visionguard.preprocess import decode_and_resize, to_clean_jpeg


def png_bytes(mode='RGBA', size=(64, 48)):
    buf = io.BytesIO()
    Image.new(mode, size, (255, 0, 0, 128) if mode == 'RGBA' else (255, 0, 0)).save(buf, format='PNG')
    return buf.getvalue()


def test_rgba_png_becomes_rgb_jpeg():
    with Image.open(io.BytesIO(to_clean_jpeg(png_bytes()))) as im:
        assert (im.format, im.mode, im.size) == ('JPEG', 'RGB', (64, 48))


def test_non_image_is_rejected():
    with pytest.raises(ValueError):
        to_clean_jpeg(b'hello, I am not an image')


def test_tiny_image_is_rejected():
    with pytest.raises(ValueError):
        to_clean_jpeg(png_bytes('RGB', (10, 10)))


def test_resize_gives_model_shape():
    assert tuple(decode_and_resize(to_clean_jpeg(png_bytes()), 224).shape) == (224, 224, 3)