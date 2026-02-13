"""
Utility for correctly formatting image content blocks for the Anthropic Messages API.

The Anthropic API requires a `media_type` field when sending base64-encoded images.
Omitting it produces:
    400 invalid_request_error: messages.X.content.Y.image.source.base64.media_type: Field required

This module provides helpers that automatically detect and include the media_type.

Usage:
    from anthropic_image_utils import make_image_content_block, fix_messages

    # === Option 1: Build image blocks correctly from the start ===

    block = make_image_content_block(file_path="screenshot.png")

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

    # === Option 2: Fix existing messages that are missing media_type ===

    messages = [...]  # your existing messages list
    fixed = fix_messages(messages)  # patches any image blocks missing media_type
    response = client.messages.create(model="...", max_tokens=1024, messages=fixed)

    # === Option 3: Validate messages before sending ===

    errors = validate_messages(messages)
    if errors:
        print("Problems found:", errors)
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


# --- Magic number signatures for media type detection ---
_MAGIC_BYTES = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WebP starts with RIFF....WEBP
]


def _detect_media_type_from_base64(base64_data: str) -> str:
    """Detect media type by decoding the first few bytes of a base64 string.

    Returns a media type string, or "image/png" as a safe fallback.
    """
    try:
        # Only need the first ~16 bytes to check magic numbers
        # Decode a small prefix (24 base64 chars = 18 raw bytes)
        prefix = base64_data[:24]
        # Pad if needed
        padding = 4 - len(prefix) % 4
        if padding != 4:
            prefix += "=" * padding
        raw = base64.standard_b64decode(prefix)

        for magic, media_type in _MAGIC_BYTES:
            if raw[: len(magic)] == magic:
                return media_type
    except Exception:
        pass

    # Default fallback - PNG is the most common for screenshots
    return "image/png"


def fix_messages(messages: list) -> list:
    """Fix a messages array by adding missing media_type to any base64 image blocks.

    This is the quickest fix for the error:
        messages.X.content.Y.image.source.base64.media_type: Field required

    It walks through all messages and content blocks, finds image blocks with
    base64 sources that are missing media_type, and auto-detects it from the
    image data's magic bytes.

    Args:
        messages: A list of message dicts (the "messages" parameter for the API).

    Returns:
        The same list, mutated in-place with media_type added where missing.
        Also returned for convenience.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image":
                continue
            source = block.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") != "base64":
                continue
            # This is a base64 image block - ensure media_type exists
            if not source.get("media_type"):
                data = source.get("data", "")
                source["media_type"] = _detect_media_type_from_base64(data)
    return messages


def validate_messages(messages: list) -> list[str]:
    """Check messages for image blocks that would cause API errors.

    Returns a list of human-readable error descriptions. Empty list = all good.
    """
    errors = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image":
                continue
            source = block.get("source")
            if not isinstance(source, dict):
                errors.append(
                    f"messages[{i}].content[{j}]: image block has no 'source'"
                )
                continue
            if source.get("type") == "base64":
                if not source.get("media_type"):
                    errors.append(
                        f"messages[{i}].content[{j}]: base64 image missing "
                        f"'media_type' (add e.g. 'image/png')"
                    )
                elif source["media_type"] not in SUPPORTED_MEDIA_TYPES:
                    errors.append(
                        f"messages[{i}].content[{j}]: unsupported media_type "
                        f"'{source['media_type']}'"
                    )
                if not source.get("data"):
                    errors.append(
                        f"messages[{i}].content[{j}]: base64 image missing 'data'"
                    )
    return errors
