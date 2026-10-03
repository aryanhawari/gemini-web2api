"""Scotty resumable image upload (content-push.googleapis.com) + MIME detection.

Two-step flow:
1. POST {UPLOAD_BASE} with X-Goog-Upload-Command: start -> response header
   X-Goog-Upload-URL carries the session URL.
2. POST the bytes there with X-Goog-Upload-Command: upload, finalize ->
   response body is the file reference that goes into payload inner[0][3].

Push-ID / X-Client-Pctx come from WIZ_global_data (qKIAYe / Ylro7b) on the
Gemini page, cached for 10 minutes, with a random fallback when the page cannot
be fetched. Anonymous uploads often fail — a cookie is recommended.
"""

import random
import re
import threading
import time
import uuid

from .gemini import (UPLOAD_BASE, GeminiError, QKIAYE_RE, YLRO7B_RE)

META_TTL_SEC = 600

_MIME_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff",
    "image/avif": ".avif", "image/heic": ".heic",
}


def make_file_name(mime):
    """Random upload filename in the shape Gemini's frontend uses."""
    ext = _MIME_EXT.get(mime, ".jpg")
    return f"input_{random.randint(1000000, 9999999)}{ext}"


try:
    from curl_cffi import CurlMime  # type: ignore
    HAVE_CURL_CFFI = True
except ImportError:  # pragma: no cover
    HAVE_CURL_CFFI = False


def detect_image_mime(data: bytes):
    """Magic-byte MIME detection for the formats Gemini accepts."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand[:2] == b"av":
            return "image/avif"
        if brand[:3] in (b"hei", b"mif", b"msf") or brand[:3] == b"hev":
            return "image/heic"
    return None


class UploadError(GeminiError):
    pass


class ImageUploader:
    def __init__(self, client):
        self.client = client
        self._lock = threading.Lock()
        self._meta = None  # (push_id, pctx, fetched_at)

    def _page_meta(self):
        with self._lock:
            now = time.time()
            if self._meta and now - self._meta[2] < META_TTL_SEC:
                return self._meta[0], self._meta[1]
            push_id, pctx = None, None
            try:
                html = self.client.fetch_page("/app")
                match = QKIAYE_RE.search(html)
                if match:
                    push_id = match.group(1)
                match = YLRO7B_RE.search(html)
                if match:
                    pctx = match.group(1)
            except GeminiError:
                pass
            if not push_id:
                push_id = str(uuid.uuid4())
            self._meta = (push_id, pctx, now)
            return push_id, pctx

    def upload(self, data: bytes, mime: str = None) -> str:
        """Uploads image bytes, returns the file reference for the payload.

        Current upstream format: a single multipart POST to content-push with
        Push-ID + X-Tenant-Id headers; the response body IS the reference.
        """
        if not mime:
            mime = detect_image_mime(data)
            if not mime:
                raise UploadError("unsupported image format (magic bytes not recognised)")
        push_id, _ = self._page_meta()
        filename = make_file_name(mime)

        headers = self.client._common_headers()
        headers.update({
            "Origin": "https://gemini.google.com",
            "Referer": "https://gemini.google.com/",
            "X-Tenant-Id": "bard-storage",
            "Push-ID": push_id,
        })

        sess = self.client._curl()
        if sess is not None:
            # Chrome TLS fingerprint — Google is strict about upload requests
            mime_part = CurlMime()
            mime_part.addpart(name="file", content_type=mime,
                              filename=filename, data=data)
            resp = sess.post(UPLOAD_BASE, multipart=mime_part, headers=headers)
        else:
            http = self.client._http()
            if http is None:
                raise UploadError(
                    "image upload requires httpx and works best with curl_cffi "
                    "(pip install curl_cffi httpx)")
            resp = http.post(UPLOAD_BASE, files={"file": (filename, data, mime)},
                             headers=headers)
        if resp.status_code != 200:
            raise UploadError(f"upload failed: HTTP {resp.status_code}")
        ref = resp.text.strip()
        if not ref:
            raise UploadError("upload returned an empty file reference")
        return ref
