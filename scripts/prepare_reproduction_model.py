#!/usr/bin/env python3
"""Download the pinned official Qwen checkpoint, resume, and verify every file.

Only standard-library dependencies are used. No model service or GPU is started.
Run this on H100 so 65 GB of weights are not copied through the Windows disk.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

LOCK = threading.Lock()


def log(event: str, **values: object) -> None:
    with LOCK:
        print(json.dumps({"time": time.time(), "event": event, **values}), flush=True)


def valid(path: Path, item: dict) -> bool:
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    if item.get("sha256"):
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {item['bytes']}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    expected = item.get("sha256") or item["git_blob_sha1"]
    return digest.hexdigest() == expected


def serve_relay(manifest: dict, port: int) -> None:
    """Serve only pinned manifest paths from the tested direct mirror route.

    The relay streams bytes without a disk cache. CDN redirects remain local;
    the remote downloader still checks every final file against official hashes.
    """
    prefix = f"/{manifest['repository']}/resolve/{manifest['revision']}/"
    allowed = {prefix + f["path"]: f for f in manifest["files"]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            # Avoid printing arbitrary request paths or signed redirect URLs.
            return

        def do_GET(self) -> None:
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            item = allowed.get(path)
            if item is None:
                self.send_error(404, "Path is not in the pinned model manifest")
                return
            size = item["bytes"]
            range_header = self.headers.get("Range")
            start, end = 0, size - 1
            if range_header:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if not match:
                    self.send_error(416, "Only a single explicit byte range is supported")
                    return
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                if start > end or start >= size:
                    self.send_error(416, "Requested byte range exceeds manifest size")
                    return
            headers = {"User-Agent": "smarthome-pinned-model-relay/1.0"}
            if range_header:
                headers["Range"] = f"bytes={start}-{end}"
            origin = "https://hf-mirror.com" + urllib.parse.quote(path, safe="/") + f"?download=true&attempt={time.time_ns()}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            written = 0
            headers_sent = False
            try:
                with opener.open(urllib.request.Request(origin, headers=headers), timeout=120) as response:
                    expected_status = 206 if range_header else 200
                    expected_range = f"bytes {start}-{end}/{size}"
                    expected_length = end - start + 1
                    if response.status != expected_status:
                        raise RuntimeError("Origin response status does not match requested range")
                    if range_header and response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Origin Content-Range does not match manifest and requested range")
                    origin_length = response.headers.get("Content-Length")
                    if origin_length is not None and int(origin_length) != expected_length:
                        raise RuntimeError("Origin Content-Length does not match manifest")
                    self.send_response(expected_status)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(expected_length))
                    self.send_header("Accept-Ranges", "bytes")
                    if range_header:
                        self.send_header("Content-Range", expected_range)
                    self.end_headers()
                    headers_sent = True
                    while written < expected_length:
                        chunk = response.read(min(1024 * 1024, expected_length - written))
                        if not chunk:
                            raise RuntimeError("Origin stream ended before declared length")
                        self.wfile.write(chunk)
                        written += len(chunk)
                    self.wfile.flush()
                log("relay_served", file=item["path"], start=start, end=end, bytes=written)
            except Exception as exc:
                log("relay_failed", file=item["path"], bytes=written, error=type(exc).__name__)
                if not headers_sent:
                    self.send_error(502, "Pinned model origin request failed")
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    log("relay_listening", address="127.0.0.1", port=port, repository=manifest["repository"], revision=manifest["revision"], files=len(allowed))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def download(item: dict, manifest: dict, args: argparse.Namespace) -> None:
    target = args.destination / item["path"]
    target.resolve().relative_to(args.destination.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid(target, item):
        log("already_verified", file=item["path"], bytes=item["bytes"])
        return
    if target.exists():
        raise RuntimeError(f"Existing final file failed validation: {target}")
    partial = target.with_name(target.name + ".partial")
    url = f"{args.endpoint.rstrip('/')}/{manifest['repository']}/resolve/{manifest['revision']}/{urllib.parse.quote(item['path'])}"
    handler = urllib.request.ProxyHandler({"https": args.proxy, "http": args.proxy} if args.proxy else {})
    opener = urllib.request.build_opener(handler)
    for attempt in range(1, args.retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item["bytes"]:
                if not valid(partial, item):
                    raise RuntimeError(f"Complete partial file failed hash: {partial}")
                partial.replace(target)
                log("verified", file=item["path"], bytes=item["bytes"])
                return
            if offset > item["bytes"]:
                raise RuntimeError(f"Oversized partial file: {partial}")
            headers = {"User-Agent": "smarthome-agent-rl-model-preparation/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            # Redirect URLs can expire; each attempt resolves a fresh signed URL.
            # Bypass stale redirect caches after a signed CDN URL expires.
            request_url = url + f"?download=true&resume_offset={offset}&attempt={time.time_ns()}"
            request = urllib.request.Request(request_url, headers=headers)
            with opener.open(request, timeout=args.timeout) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor Range; preserve partial instead of overwriting it")
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                previous = time.monotonic()
                position = offset
                with partial.open("ab" if offset else "wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        if position + len(chunk) > item["bytes"]:
                            raise RuntimeError("Response exceeds manifest size")
                        handle.write(chunk)
                        position += len(chunk)
                        if time.monotonic() - previous >= 20:
                            handle.flush()
                            log("progress", file=item["path"], bytes=position, total=item["bytes"])
                            previous = time.monotonic()
                if position != item["bytes"]:
                    raise RuntimeError(f"Incomplete response: {position}/{item['bytes']}")
            if not valid(partial, item):
                raise RuntimeError(f"Downloaded file failed hash: {partial}")
            partial.replace(target)
            log("verified", file=item["path"], bytes=item["bytes"])
            return
        except RuntimeError as exc:
            # Hash mismatch, unsupported ranges, or manifest inconsistency need
            # inspection; repeated retries would just reread gigabytes.
            log("failed", file=item["path"], attempt=attempt, error=str(exc))
            raise
        except Exception as exc:
            log("retry", file=item["path"], attempt=attempt, error=str(exc))
            if attempt == args.retries:
                raise
            time.sleep(min(30, 2 ** min(attempt, 5)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reproduction-model-qwen3-32b.json")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--serve-relay", action="store_true")
    parser.add_argument("--listen-port", type=int, default=18099)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if args.serve_relay:
        if args.verify_only or args.destination:
            parser.error("Relay mode does not use a destination or verify-only")
        serve_relay(manifest, args.listen_port)
        return
    if args.destination is None:
        parser.error("--destination is required for download or verification")
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.destination.mkdir(parents=True, exist_ok=True)
    log("manifest", repository=manifest["repository"], revision=manifest["revision"], total=manifest["total_bytes"], destination=str(args.destination))
    if not args.verify_only:
        remaining = sum(max(0, f["bytes"] - ((args.destination / (f["path"] + ".partial")).stat().st_size if (args.destination / (f["path"] + ".partial")).exists() else (args.destination / f["path"]).stat().st_size if (args.destination / f["path"]).exists() else 0)) for f in manifest["files"])
        if shutil.disk_usage(args.destination).free < remaining + 1024 ** 3:
            raise RuntimeError("Insufficient free space for pinned checkpoint and 1 GiB margin")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, f, manifest, args) for f in manifest["files"]]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    invalid = [f["path"] for f in manifest["files"] if not valid(args.destination / f["path"], f)]
    result = {"repository": manifest["repository"], "revision": manifest["revision"], "verified": not invalid, "invalid_or_missing": invalid, "checked_at": time.time()}
    (args.destination / "download-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log("finished", **result)
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
