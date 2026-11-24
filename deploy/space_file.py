"""Emit a one-file Space deployment pinned to an already-published GitHub commit."""
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    sha = sys.argv[1]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit("Pass a full Git commit SHA.")
    runtime = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    print(f"""FROM alpine:3.21 AS source
ADD https://codeload.github.com/avish006/Rag-Project/tar.gz/{sha} /tmp/source.tgz
RUN mkdir /source && tar -xzf /tmp/source.tgz -C /source --strip-components=1
""")
    for line in runtime.splitlines():
        if line.startswith("COPY "):
            *inputs, target = line[5:].split()
            line = "COPY --from=source " + " ".join("/source/" + f for f in inputs) + " " + target
        print(line)
