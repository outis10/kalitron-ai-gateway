from fastapi import HTTPException, UploadFile, status

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
READ_CHUNK_SIZE = 1024 * 1024


def validate_image_file(file: UploadFile) -> None:
    """Raise HTTPException if the uploaded file has an unsupported content type."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WebP.",
        )


async def read_limited_upload(
    file: UploadFile, max_file_bytes: int, max_file_size_mb: int
) -> bytes:
    """
    Read an uploaded file in bounded chunks so oversized payloads are rejected early.

    The downstream AI provider still needs the full image bytes, but this avoids reading
    arbitrarily large uploads into memory before enforcing the configured size limit.
    """
    total_bytes = 0
    chunks: list[bytes] = []

    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_file_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_file_size_mb} MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)
