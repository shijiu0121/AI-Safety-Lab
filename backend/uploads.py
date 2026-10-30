"""统一图片限制，既供预览也供实时防护使用。"""

import io
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException


async def read_image(upload):
    payload = await upload.read(12 * 1024 * 1024 + 1)
    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(413, "图片不能超过 12 MB。")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 16000000:
                raise ValueError("图片不能超过 1600 万像素。")
            image.load()
            information = dict(
                width=image.width,
                height=image.height,
                format=image.format,
                bytes=len(payload),
            )
            return image.convert("RGB"), information
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(422, "无法读取图片：" + str(exc)) from exc
