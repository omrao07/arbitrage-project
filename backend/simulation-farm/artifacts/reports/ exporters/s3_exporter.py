# simulation-farm/artifacts/reports/exporters/s3_exporter.py
"""
S3 Exporter for simulation reports.

Features
--------
- Upload file/bytes/directory to S3 with a configurable key prefix
- Optional gzip for text-like assets (HTML/CSS/JS/JSON/SVG)
- Sets Content-Type, Cache-Control; optional public-read ACL
- Create bucket (optional), and generate presigned GET URLs
- Fallback: upload via a provided pre-signed PUT URL if boto3 isn't installed

Usage
-----
exp = S3Exporter(bucket="my-bucket", prefix="reports/run_123/", region="us-east-1",
                 public=False, gzip_text=True)
exp.upload_file("artifacts/reports/run_123/summary.html", content_type="text/html")
exp.upload_dir("artifacts/reports/run_123")
url = exp.generate_presigned_url("summary.html", ttl_seconds=3600)
"""

from __future__ import annotations

import gzip
import io
import mimetypes
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Optional deps
try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
    _HAVE_BOTO3 = True
except Exception:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    _HAVE_BOTO3 = False

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


@dataclass
class S3ExporterConfig:
    bucket: str
    prefix: str = ""                 # e.g., "reports/run_001/"
    region: Optional[str] = None
    public: bool = False             # if True, set ACL public-read on upload
    default_cache_control: str = "public, max-age=300"
    gzip_text: bool = False          # gzip text-like objects before upload
    sse: Optional[str] = None        # e.g. "AES256" or "aws:kms"
    sse_kms_key_id: Optional[str] = None  # if using KMS
    endpoint_url: Optional[str] = None    # for S3-compatible stores (MinIO, etc.)


class S3Exporter:
    def __init__(self,
                 bucket: str,
                 prefix: str = "",
                 *,
                 region: Optional[str] = None,
                 public: bool = False,
                 default_cache_control: str = "public, max-age=300",
                 gzip_text: bool = False,
                 sse: Optional[str] = None,
                 sse_kms_key_id: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        # normalize prefix to "foo/bar/" or ""
        pfx = prefix.strip("/")
        if pfx and not pfx.endswith("/"):
            pfx += "/"
        self.cfg = S3ExporterConfig(
            bucket=bucket,
            prefix=pfx,
            region=region,
            public=public,
            default_cache_control=default_cache_control,
            gzip_text=gzip_text,
            sse=sse,
            sse_kms_key_id=sse_kms_key_id,
            endpoint_url=endpoint_url,
        )
        self._s3 = None
        self._client = None
        if _HAVE_BOTO3:
            self._s3 = boto3.resource("s3", region_name=self.cfg.region, endpoint_url=self.cfg.endpoint_url) # type: ignore
            self._client = boto3.client("s3", region_name=self.cfg.region, endpoint_url=self.cfg.endpoint_url) # type: ignore

    # ----------------- public API -----------------

    def upload_file(self, path: str, dest: Optional[str] = None, *,
                    content_type: Optional[str] = None,
                    cache_control: Optional[str] = None,
                    gzip_override: Optional[bool] = None) -> str:
        src = pathlib.Path(path)
        if not src.exists():
            raise FileNotFoundError(path)
        key = self._dest_key(dest or src.name)
        ct = content_type or _guess_content_type(src.name)
        cc = cache_control or self.cfg.default_cache_control

        data = src.read_bytes()
        do_gzip = self._should_gzip(src.name, gzip_override)
        if do_gzip:
            data = _gzip_bytes(data)
        self._put_bytes(key, data, content_type=ct, cache_control=cc, gzip_applied=do_gzip)
        return f"s3://{self.cfg.bucket}/{key}"

    def upload_bytes(self, data: bytes, dest: str, *,
                     content_type: str = "application/octet-stream",
                     cache_control: Optional[str] = None,
                     gzip_override: Optional[bool] = None) -> str:
        key = self._dest_key(dest)
        cc = cache_control or self.cfg.default_cache_control
        do_gzip = self._should_gzip(dest, gzip_override)
        if do_gzip:
            data = _gzip_bytes(data)
        self._put_bytes(key, data, content_type=content_type, cache_control=cc, gzip_applied=do_gzip)
        return f"s3://{self.cfg.bucket}/{key}"

    def upload_dir(self, directory: str, *, strip_prefix: Optional[str] = None) -> List[str]:
        base = pathlib.Path(directory)
        if not base.is_dir():
            raise NotADirectoryError(directory)
        strip = pathlib.Path(strip_prefix) if strip_prefix else base
        urls: List[str] = []
        for fp in base.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(strip).as_posix()
                urls.append(self.upload_file(str(fp), dest=rel))
        return urls

    def ensure_bucket(self) -> None:
        if not _HAVE_BOTO3:
            raise RuntimeError("ensure_bucket requires boto3. Install with: pip install boto3")
        try:
            self._client.head_bucket(Bucket=self.cfg.bucket) # type: ignore
        except ClientError:
            create_args = {"Bucket": self.cfg.bucket}
            # us-east-1 doesn't accept LocationConstraint
            if self.cfg.region and self.cfg.region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {"LocationConstraint": self.cfg.region} # type: ignore
            self._client.create_bucket(**create_args) # type: ignore

    def generate_presigned_url(self, object_path: str, *, ttl_seconds: int = 3600, method: str = "get_object") -> str:
        """
        method: 'get_object' or 'put_object'
        """
        key = self._dest_key(object_path)
        if not _HAVE_BOTO3:
            raise RuntimeError("Presigned URLs require boto3")
        return self._client.generate_presigned_url( # type: ignore
            ClientMethod=method,
            Params={"Bucket": self.cfg.bucket, "Key": key},
            ExpiresIn=int(ttl_seconds),
        )

    # ----------------- internals -----------------

    def _dest_key(self, name: str) -> str:
        n = name.lstrip("/").replace("\\", "/")
        return self.cfg.prefix + n

    def _should_gzip(self, name: str, override: Optional[bool]) -> bool:
        if override is not None:
            return bool(override)
        if not self.cfg.gzip_text:
            return False
        ct = _guess_content_type(name) or ""
        return ct.startswith("text/") or ct in ("application/json", "application/javascript", "image/svg+xml")

    def _put_bytes(self, key: str, data: bytes, *, content_type: str, cache_control: str, gzip_applied: bool) -> None:
        if _HAVE_BOTO3:
            extra = {
                "ContentType": content_type,
                "CacheControl": cache_control,
            }
            if gzip_applied:
                extra["ContentEncoding"] = "gzip"
            if self.cfg.public:
                extra["ACL"] = "public-read"
            if self.cfg.sse:
                extra["ServerSideEncryption"] = self.cfg.sse
                if self.cfg.sse == "aws:kms" and self.cfg.sse_kms_key_id:
                    extra["SSEKMSKeyId"] = self.cfg.sse_kms_key_id
            obj = self._s3.Object(self.cfg.bucket, key) # type: ignore
            obj.put(Body=data, **extra)
            return

        # ---- fallback via pre-signed PUT URL ----
        if requests is None:
            raise RuntimeError("boto3 not installed and 'requests' missing for fallback HTTP upload")
        put_url = os.getenv("S3_SIGNED_PUT_URL")
        if not put_url:
            raise RuntimeError("Set S3_SIGNED_PUT_URL to a pre-signed PUT URL to use fallback uploader")
        headers = {"Content-Type": content_type, "Cache-Control": cache_control}
        if gzip_applied:
            headers["Content-Encoding"] = "gzip"
        r = requests.put(put_url, data=data, headers=headers, timeout=60)
        r.raise_for_status()

# ----------------- helpers -----------------

def _guess_content_type(name: str) -> str:
    ct, _ = mimetypes.guess_type(name)
    return ct or "application/octet-stream"

def _gzip_bytes(data: bytes) -> bytes:
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(data)
    return bio.getvalue()


# ----------------- CLI (optional) -----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Upload files/dirs to S3")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--region", default=None)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--gzip-text", action="store_true")
    ap.add_argument("--endpoint-url", default=None, help="Custom S3 endpoint (MinIO, etc.)")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    exp = S3Exporter(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        public=args.public,
        gzip_text=args.gzip_text,
        endpoint_url=args.endpoint_url,
    )

    urls: List[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            urls.extend(exp.upload_dir(p))
        else:
            urls.append(exp.upload_file(p))
    for u in urls:
        print(u)