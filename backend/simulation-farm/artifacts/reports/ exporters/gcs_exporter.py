# simulation-farm/artifacts/reports/exporters/gcs_exporter.py
"""
GCS Exporter for simulation reports.

Features
--------
- Upload files/bytes/directories to a bucket (prefix-aware)
- Optional gzip on-the-fly for text assets
- Cache-Control & Content-Type handling
- Make objects public or generate signed URLs
- Works with ADC or a service account JSON
- Fallback minimal signed-URL uploader if google-cloud-storage isn't installed

Usage
-----
exp = GCSExporter(bucket="my-bucket", prefix="reports/", make_public=False)
exp.upload_file("artifacts/reports/run123/summary.html", content_type="text/html", cache_control="public,max-age=300")
url = exp.generate_signed_url("run123/summary.html", ttl_seconds=3600)

# Upload a whole directory
exp.upload_dir("artifacts/reports/run123")  # preserves relative paths under prefix
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import mimetypes
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, Optional

# Optional dependency: google-cloud-storage
try:
    from google.cloud import storage  # type: ignore
    _HAVE_GCS = True
except Exception:
    storage = None  # type: ignore
    _HAVE_GCS = False

# Optional dependency for fallback signed URL flow
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


@dataclass
class GCSExporterConfig:
    bucket: str
    prefix: str = ""                  # e.g., "reports/"
    make_public: bool = False         # if True, set blobs to public on upload
    default_cache_control: str = "public, max-age=300"
    gzip_text: bool = False           # gzip text-like content on upload
    credentials_path: Optional[str] = None  # path to service account json (else ADC)
    project: Optional[str] = None     # GCP project (optional)


class GCSExporter:
    def __init__(self, bucket: str, prefix: str = "", *, make_public: bool = False,
                 default_cache_control: str = "public, max-age=300",
                 gzip_text: bool = False,
                 credentials_path: Optional[str] = None,
                 project: Optional[str] = None):
        self.cfg = GCSExporterConfig(
            bucket=bucket,
            prefix=prefix.strip("/")+("/" if prefix and not prefix.endswith("/") else ""),
            make_public=make_public,
            default_cache_control=default_cache_control,
            gzip_text=gzip_text,
            credentials_path=credentials_path,
            project=project,
        )
        self._client = None
        self._bucket = None
        if _HAVE_GCS:
            self._client = self._make_client()
            self._bucket = self._client.bucket(self.cfg.bucket)

    # ---------------------- public API ----------------------

    def upload_file(self, path: str, dest: Optional[str] = None, *,
                    content_type: Optional[str] = None,
                    cache_control: Optional[str] = None,
                    gzip_override: Optional[bool] = None) -> str:
        """
        Upload a single file. Returns the gs:// URL.
        """
        src = pathlib.Path(path)
        if not src.exists():
            raise FileNotFoundError(path)
        name = dest or src.name
        key = self.cfg.prefix + name
        ct = content_type or _guess_content_type(src.name)
        cc = cache_control or self.cfg.default_cache_control

        with open(src, "rb") as f:
            data = f.read()

        # Decide gzip
        do_gzip = self._should_gzip(src.name, gzip_override)
        if do_gzip:
            data = _gzip_bytes(data)
        return self._put_bytes(key, data, content_type=ct, cache_control=cc, gzip_applied=do_gzip)

    def upload_bytes(self, data: bytes, dest: str, *,
                     content_type: str = "application/octet-stream",
                     cache_control: Optional[str] = None,
                     gzip_override: Optional[bool] = None) -> str:
        do_gzip = self._should_gzip(dest, gzip_override)
        if do_gzip:
            data = _gzip_bytes(data)
        cc = cache_control or self.cfg.default_cache_control
        key = self.cfg.prefix + dest.lstrip("/")
        return self._put_bytes(key, data, content_type=content_type, cache_control=cc, gzip_applied=do_gzip)

    def upload_dir(self, directory: str, *, strip_prefix: Optional[str] = None) -> list[str]:
        """
        Upload all files under `directory`. Returns a list of gs:// URLs.
        `strip_prefix` lets you trim the source path from destination keys.
        """
        base = pathlib.Path(directory)
        if not base.is_dir():
            raise NotADirectoryError(directory)

        urls: list[str] = []
        for fp in base.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(strip_prefix or base)
                dest = str(rel).replace(os.sep, "/")
                urls.append(self.upload_file(str(fp), dest=dest))
        return urls

    def ensure_bucket(self, *, location: str = "US", storage_class: str = "STANDARD") -> None:
        if not _HAVE_GCS:
            raise RuntimeError("ensure_bucket requires google-cloud-storage. Install with: pip install google-cloud-storage")
        if self._bucket.exists(): # type: ignore
            return
        self._client.create_bucket(self._bucket, location=location, storage_class=storage_class) # type: ignore

    def generate_signed_url(self, object_path: str, *, ttl_seconds: int = 3600, method: str = "GET") -> str:
        """
        Generate a signed URL for an object path (relative to exporter prefix).
        """
        key = self.cfg.prefix + object_path.lstrip("/")
        if _HAVE_GCS:
            blob = self._bucket.blob(key) # type: ignore
            return blob.generate_signed_url(expiration=ttl_seconds, method=method)
        # fallback: cannot sign without credentials lib; raise a friendly error
        raise RuntimeError("Signed URLs require google-cloud-storage (or implement fallback signer).")

    # ---------------------- internals ----------------------

    def _make_client(self):
        if self.cfg.credentials_path:
            from google.oauth2 import service_account  # type: ignore
            creds = service_account.Credentials.from_service_account_file(self.cfg.credentials_path)
            return storage.Client(project=self.cfg.project, credentials=creds) # type: ignore
        return storage.Client(project=self.cfg.project) # type: ignore

    def _should_gzip(self, name: str, override: Optional[bool]) -> bool:
        if override is not None:
            return bool(override)
        if not self.cfg.gzip_text:
            return False
        ct = _guess_content_type(name) or ""
        return ct.startswith("text/") or ct in ("application/json", "application/javascript", "image/svg+xml")

    def _put_bytes(self, key: str, data: bytes, *, content_type: str, cache_control: str, gzip_applied: bool) -> str:
        if _HAVE_GCS:
            blob = self._bucket.blob(key) # type: ignore
            blob.cache_control = cache_control
            blob.content_type = content_type
            if gzip_applied:
                blob.content_encoding = "gzip"
            blob.upload_from_file(io.BytesIO(data), rewind=True, size=len(data), content_type=content_type)
            if self.cfg.make_public:
                try:
                    blob.make_public()
                except Exception:
                    pass
            return f"gs://{self.cfg.bucket}/{key}"

        # ---- minimal fallback via signed URL upload (requires requests and env SIGNED_PUT_URL) ----
        if requests is None:
            raise RuntimeError("google-cloud-storage not installed and no 'requests' for fallback HTTP upload")
        signed_put_url = os.getenv("GCS_SIGNED_PUT_URL")
        if not signed_put_url:
            raise RuntimeError("Set GCS_SIGNED_PUT_URL env var to a pre-signed PUT URL when library is unavailable")
        headers = {"Content-Type": content_type, "Cache-Control": cache_control}
        if gzip_applied:
            headers["Content-Encoding"] = "gzip"
        r = requests.put(signed_put_url, data=data, headers=headers, timeout=30)
        r.raise_for_status()
        return signed_put_url.split("?")[0]  # object URL without query


# ---------------------- helpers ----------------------

def _guess_content_type(name: str) -> str:
    ct, _ = mimetypes.guess_type(name)
    return ct or "application/octet-stream"

def _gzip_bytes(data: bytes) -> bytes:
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(data)
    return bio.getvalue()


# ---------------------- CLI (optional) ----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Upload files/dirs to GCS")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--gzip-text", action="store_true")
    ap.add_argument("--credentials", default=None, help="Path to service account JSON (else ADC)")
    ap.add_argument("--project", default=None)
    ap.add_argument("paths", nargs="+", help="Files or directories to upload")
    args = ap.parse_args()

    exp = GCSExporter(
        bucket=args.bucket,
        prefix=args.prefix,
        make_public=args.public,
        gzip_text=args.gzip_text,
        credentials_path=args.credentials,
        project=args.project,
    )

    uploaded: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            uploaded.extend(exp.upload_dir(p))
        else:
            uploaded.append(exp.upload_file(p))

    for u in uploaded:
        print(u)