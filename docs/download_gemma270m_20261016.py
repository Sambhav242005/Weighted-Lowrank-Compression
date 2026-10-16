"""Explicit pinned model download, independent of torch; resumable partial files."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
import uuid
import traceback

REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
FILES = ["config.json", "generation_config.json", "special_tokens_map.json",
         "added_tokens.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "model.safetensors"]


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def main():
    out = Path("results/gemma270m-download") / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    target = Path("paper-assets") / ("gemma270m-" + REVISION)
    target.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(root / "token")))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and token_path.exists():
        token = token_path.read_text().strip()
    opener = urllib.request.build_opener(SafeRedirect())
    state = {"state": "running", "model": "google/gemma-3-270m", "revision": REVISION,
             "destination": str(target.resolve()), "files": {}}

    def status():
        (out / "status.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    status()
    try:
        for filename in FILES:
            dest = target / filename
            partial = target / (filename + ".partial")
            if not dest.exists():
                for attempt in range(1, 6):
                    offset = partial.stat().st_size if partial.exists() else 0
                    request = urllib.request.Request(f"https://huggingface.co/google/gemma-3-270m/resolve/{REVISION}/{filename}")
                    if token:
                        request.add_header("Authorization", "Bearer " + token)
                    if offset:
                        request.add_header("Range", f"bytes={offset}-")
                    try:
                        with opener.open(request, timeout=120) as response:
                            if offset and response.status != 206:
                                raise RuntimeError("Server refused resume; partial file retained")
                            expected = int(response.headers.get("Content-Length", 0)) + offset
                            print(f"Downloading {filename}, offset {offset}, expected {expected}", flush=True)
                            with partial.open("ab" if offset else "xb") as stream:
                                last_report = time.monotonic()
                                while True:
                                    block = response.read(1024 * 1024)
                                    if not block:
                                        break
                                    stream.write(block)
                                    if time.monotonic() - last_report >= 15:
                                        stream.flush()
                                        state["active_file"] = filename
                                        state["downloaded_bytes"] = stream.tell()
                                        state["expected_bytes"] = expected
                                        status()
                                        print(f"{filename}: {stream.tell()}/{expected} bytes", flush=True)
                                        last_report = time.monotonic()
                            if expected and partial.stat().st_size != expected:
                                raise OSError("Incomplete response")
                        partial.rename(dest)
                        break
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        print(f"Attempt {attempt}: {type(exc).__name__}; partial file preserved", flush=True)
                        if attempt == 5:
                            raise
                        time.sleep(2)
            digest = hashlib.sha256()
            with dest.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            state["files"][filename] = {"bytes": dest.stat().st_size, "sha256": digest.hexdigest()}
            status()
            print(f"Verified local file: {filename}, {dest.stat().st_size} bytes", flush=True)
        state["state"] = "completed"
        status()
        print("Model download completed:", target.resolve(), flush=True)
    except BaseException:
        state["state"] = "failed"
        status()
        (out / "stderr.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
