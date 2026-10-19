"""Keep completed installer downloads for reuse; hard links do not duplicate data."""
import email
import os
from pathlib import Path
import zipfile
import tarfile

root = Path("/mnt/data/home/zzr")
dest = root / "cache/wheels"
dest.mkdir(parents=True, exist_ok=True)
count = 0
for path in (root / "tmp/atod_setup").glob("pip-unpack-*/*.whl"):
    try:
        with zipfile.ZipFile(path) as wheel:
            name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
            metadata = email.message_from_bytes(wheel.read(name))
            tags = "-".join(path.name[:-4].split("-")[-3:])
            target = dest / f"{metadata['Name'].replace('-', '_')}-{metadata['Version']}-{tags}.whl"
        if not target.exists():
            os.link(path, target)
            count += 1
    except (OSError, zipfile.BadZipFile, StopIteration):
        pass
print(f"Cached {count} additional completed wheels")
sources = 0
for path in (root / "tmp/atod_setup").glob("pip-unpack-*/*.tar.gz"):
    target = dest / path.name
    if target.exists():
        continue
    try:
        with tarfile.open(path, "r:gz") as archive:
            archive.getmembers()
        os.link(path, target)
        sources += 1
    except (OSError, tarfile.TarError, EOFError):
        pass
print(f"Cached {sources} additional source archives")
