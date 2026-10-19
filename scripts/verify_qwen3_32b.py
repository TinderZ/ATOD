"""Offline size/checksum/header verification of the pinned official model."""
import hashlib
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/mnt/data/home/zzr/models/Qwen3-32B')
FILES = ROOT / 'configs/setup/qwen3_32b_remote_files.json'


def main():
    started = time.monotonic()
    items = json.loads(FILES.read_text())
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    expected_tensors = index['weight_map']
    observed_tensors = {}
    verified_files = []
    for item in items:
        assert item['type'] == 'file'
        path = MODEL / item['path']
        size = path.stat().st_size
        assert size == item['size'], (path.name, size, item['size'])
        if item.get('lfs'):
            expected = item['lfs']['oid']
            digest = hashlib.sha256()
        else:
            expected = item['oid']
            digest = hashlib.sha1()
            digest.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, f'Checksum mismatch: {path}'
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                header = json.loads(stream.read(length))
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                assert name not in observed_tensors
                assert info['dtype'] == 'BF16', (name, info['dtype'])
                assert 0 <= info['data_offsets'][0] <= info['data_offsets'][1] <= size - length - 8
                observed_tensors[name] = path.name
        verified_files.append({'file': path.name, 'size': size, 'checksum': expected})
        print(f'VERIFIED {path.name} {size}', flush=True)
    assert observed_tensors == expected_tensors, 'Weight index/header mismatch'
    result = {'repo': 'Qwen/Qwen3-32B', 'revision': '9216db5781bf21249d130ec9da846c4624c16137',
              'model': str(MODEL), 'verified_files': verified_files,
              'tensor_count': len(observed_tensors), 'all_tensors_bfloat16': True,
              'tensor_bytes': index['metadata']['total_size'],
              'elapsed_seconds': time.monotonic() - started}
    target = MODEL / 'local_verification.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2)
    print('ALL FILES AND TENSOR HEADERS VERIFIED', flush=True)


if __name__ == '__main__':
    main()
