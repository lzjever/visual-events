from __future__ import annotations

import struct


def jpeg_with_sof0_dimensions(*, width: int = 1280, height: int = 720) -> bytes:
    sof0_payload = (
        b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03"
        + b"\x01\x11\x00"
        + b"\x02\x11\x00"
        + b"\x03\x11\x00"
    )
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(sof0_payload) + 2)
        + sof0_payload
        + b"\xff\xd9"
    )


JPEG_1280X720 = jpeg_with_sof0_dimensions()
