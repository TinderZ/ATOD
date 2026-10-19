"""Bounded HTTP Range fallback using pinned Hub metadata and existing partials.

Only run after the original hf downloader has exited. Never uses credentials or
proxy settings; final files are checked before reuse and partials are retained.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/mnt/data/home/zzr/models/Qwen3-32B')
REV = '9216db5781bf21249d130ec9da846c4624c16137'
ITEMS = json.loads((ROOT / 'configs/setup/qwen3_32b_remote_files.json').read_text())
MS_FILES = json.loads((ROOT / 'configs/setup/qwen3_32b_modelscope_remote_files.json').read_text())['Data']['Files']
MS_BY_PATH = {item['Path']: item for item in MS_FILES}
WEIGHT_SOURCE = 'hf'


def expected_digest(item):
    return item['lfs']['oid'] if item.get('lfs') else item['oid']


def check(path, item):
    if path.stat().st_size != item['size']:
        return False
    h = hashlib.sha256() if item.get('lfs') else hashlib.sha1()
    if not item.get('lfs'):
        h.update(f"blob {item['size']}\0".encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest() == expected_digest(item)


def download(item):
    name = item['path']
    assert item['type'] == 'file' and '/' not in name
    path = DEST / name
    if path.exists():
        assert check(path, item), f'Existing file checksum mismatch: {path}'
        print(f'REUSE VERIFIED {name}', flush=True)
        return
    matches = list((DEST / '.cache/huggingface/download').glob(f'*.{expected_digest(item)}.incomplete'))
    assert len(matches) <= 1
    partial = matches[0] if matches else path.with_name(path.name + '.incomplete')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Separate resolver cache key from earlier HEAD/range probe requests.
    url = f'https://hf-mirror.com/Qwen/Qwen3-32B/resolve/{REV}/{name}?download=true'
    if WEIGHT_SOURCE == 'modelscope' and name.endswith('.safetensors'):
        alternative = MS_BY_PATH[name]
        assert alternative['Sha256'] == expected_digest(item) and alternative['Size'] == item['size']
        url = f"https://modelscope.cn/models/Qwen/Qwen3-32B/resolve/{alternative['Revision']}/{name}"
    last_report = 0
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            assert offset <= item['size'], f'Oversized partial: {partial}'
            if offset < item['size']:
                headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Encoding': 'identity',
                           'Range': f"bytes={offset}-{item['size'] - 1}"}
                with opener.open(urllib.request.Request(url, headers=headers), timeout=90) as r:
                    if r.status == 206:
                        actual = int(r.headers['Content-Range'].split()[1].split('-')[0])
                        assert actual == offset, (name, actual, offset)
                    elif offset or r.status != 200:
                        raise RuntimeError('Server did not honor resume request')
                    with partial.open('ab') as out:
                        while block := r.read(4 * 1024 * 1024):
                            out.write(block)
                            if time.monotonic() - last_report >= 30:
                                print(f"PROGRESS {name} {out.tell()}/{item['size']}", flush=True)
                                last_report = time.monotonic()
            assert check(partial, item), f'Incomplete or checksum mismatch: {name}'
            partial.rename(path)
            print(f"DOWNLOADED VERIFIED {name} {item['size']}", flush=True)
            return
        except Exception as e:
            print(f'ERROR attempt={attempt + 1} {name}: {type(e).__name__}: {str(e).split("?")[0]}', flush=True)
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-source', choices=['hf', 'modelscope'], default='hf')
    WEIGHT_SOURCE = parser.parse_args().weight_source
    print(f'WEIGHT_SOURCE={WEIGHT_SOURCE}; all files must match pinned HF checksums', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, sorted(ITEMS, key=lambda item: (item['path'].endswith('.safetensors'), item['path']))))
    print('ALL PINNED FILES DOWNLOADED AND VERIFIED', flush=True)
