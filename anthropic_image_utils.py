"""
Utility for correctly formatting image content blocks for the Anthropic Messages API.

The Anthropic API requires a `media_type` field when sending base64-encoded images.
Omitting it produces:
    400 invalid_request_error: messages.X.content.Y.image.source.base64.media_type: Field required

This module provides helpers that automatically detect and include the media_type.

Usage:
    from anthropic_image_utils import make_image_content_block

    # From a file path:
    block = make_image_content_block(file_path="screenshot.png")

    # From raw bytes:
    block = make_image_content_block(image_bytes=raw_bytes, media_type="image/png")

    # From an already-encoded base64 string:
    block = make_image_content_block(base64_data=b64_str, media_type="image/jpeg")

    # Then include it in your messages:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                block,
                {"type": "text", "text": "What's in this image?"}
            ]
        }]
    )
"""

import base64
import imghdr
import os


SUPPORTED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

# Maps imghdr detection results to MIME media types
_IMGHDR_TO_MEDIA_TYPE = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

# Maps file extensions to MIME media types
_EXT_TO_MEDIA_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def detect_media_type(image_bytes: bytes, file_path: str = None) -> str:
    """Detect the media type of an image from its bytes and/or file extension.

    Args:
        image_bytes: Raw image bytes.
        file_path: Optional file path used as a fallback for extension-based detection.

    Returns:
        A media type string like "image/png".

    Raises:
        ValueError: If the media type cannot be determined or is unsupported.
    """
    # Try magic-number detection first
    detected = imghdr.what(None, h=image_bytes)
    if detected and detected in _IMGHDR_TO_MEDIA_TYPE:
        return _IMGHDR_TO_MEDIA_TYPE[detected]

    # Fall back to file extension
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _EXT_TO_MEDIA_TYPE:
            return _EXT_TO_MEDIA_TYPE[ext]

    raise ValueError(
        f"Could not detect image media type. "
        f"imghdr returned: {detected!r}. "
        f"Provide media_type explicitly (one of: {', '.join(sorted(SUPPORTED_MEDIA_TYPES))})."
    )


def make_image_content_block(
    *,
    file_path: str = None,
    image_bytes: bytes = None,
    base64_data: str = None,
    media_type: str = None,
) -> dict:
    """Build a correctly-formatted image content block for the Anthropic Messages API.

    Provide exactly ONE of: file_path, image_bytes, or base64_data.

    Args:
        file_path: Path to an image file on disk.
        image_bytes: Raw image bytes.
        base64_data: Already base64-encoded image data string.
        media_type: Explicit media type (e.g. "image/png"). Auto-detected if not given
                     and file_path or image_bytes is provided.

    Returns:
        A dict suitable for inclusion in a messages content array:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "<base64-string>"
            }
        }

    Raises:
        ValueError: If inputs are invalid or media_type cannot be determined.
    """
    sources_given = sum(x is not None for x in [file_path, image_bytes, base64_data])
    if sources_given != 1:
        raise ValueError(
            "Provide exactly one of: file_path, image_bytes, or base64_data."
        )

    # Resolve image data to base64 string + raw bytes for detection
    raw_bytes = None

    if file_path is not None:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        base64_str = base64.standard_b64encode(raw_bytes).decode("ascii")
    elif image_bytes is not None:
        raw_bytes = image_bytes
        base64_str = base64.standard_b64encode(raw_bytes).decode("ascii")
    else:
        # base64_data provided
        base64_str = base64_data
        # Decode so we can auto-detect if needed
        if media_type is None:
            raw_bytes = base64.standard_b64decode(base64_data)

    # Resolve media_type
    if media_type is None:
        if raw_bytes is None:
            raise ValueError(
                "media_type is required when providing base64_data "
                "(pass media_type explicitly)."
            )
        media_type = detect_media_type(raw_bytes, file_path=file_path)

    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f"Unsupported media_type: {media_type!r}. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_MEDIA_TYPES))}."
        )

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64_str,
        },
    }
