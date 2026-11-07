"""Preprocessing shared by training and serving. Change it here only."""
import io

import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

JPEG_QUALITY = 95
MIN_SIDE = 32


def to_clean_jpeg(raw: bytes) -> bytes:
    """Any image Pillow can read -> clean RGB JPEG bytes. Raises ValueError otherwise."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.seek(0)                          # first frame of GIF / animated WebP
            img = ImageOps.exif_transpose(img)   # apply camera rotation tag
            if img.mode in ('RGBA', 'LA', 'P', 'PA'):
                img = img.convert('RGBA')
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.getchannel('A'))
                img = bg
            else:
                img = img.convert('RGB')
            if min(img.size) < MIN_SIDE:
                raise ValueError(f'image too small: {img.size}')
            out = io.BytesIO()
            img.save(out, format='JPEG', quality=JPEG_QUALITY)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f'not a decodable image: {exc}') from exc


def decode_and_resize(jpeg_bytes, img_size: int):
    """JPEG bytes (tensor) -> float32 tensor (img_size, img_size, 3), values 0..255."""
    img = tf.io.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return tf.clip_by_value(img, 0.0, 255.0)