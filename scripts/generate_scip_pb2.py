#!/usr/bin/env python3
"""
Regenerate reporag/indexer/scip_pb2.py from the official Sourcegraph SCIP proto.

Run when the SCIP protocol version changes:
    python scripts/generate_scip_pb2.py
"""
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SCIP_PROTO_URL = "https://raw.githubusercontent.com/sourcegraph/scip/main/scip.proto"
OUT_FILE = Path(__file__).parent.parent / "reporag" / "indexer" / "scip_pb2.py"


def main() -> None:
    print(f"Fetching {SCIP_PROTO_URL} ...")
    with urllib.request.urlopen(SCIP_PROTO_URL) as resp:  # noqa: S310
        proto_bytes = resp.read()

    with tempfile.TemporaryDirectory() as tmp:
        proto_path = Path(tmp) / "scip.proto"
        proto_path.write_bytes(proto_bytes)

        print("Generating scip_pb2.py via grpcio-tools ...")
        result = subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"--proto_path={tmp}",
                f"--python_out={OUT_FILE.parent}",
                str(proto_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("ERROR:", result.stderr)
            print("Install grpcio-tools first: pip install grpcio-tools")
            sys.exit(1)

    print(f"Written: {OUT_FILE}")
    print(f"Size:    {OUT_FILE.stat().st_size} bytes")


if __name__ == "__main__":
    main()
