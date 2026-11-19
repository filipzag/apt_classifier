import importlib.util
import struct
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ollama_text_only", Path(__file__).resolve().parents[1] / "scripts" / "ollama_text_only.py"
)
text_only = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(text_only)


def _kv_string(key, value):
    return (
        struct.pack("<Q", len(key))
        + key.encode()
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(value))
        + value.encode()
    )


def _kv_u32(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<II", 4, value)


def _write_toy_gguf(path):
    tensors = [("blk.0.w", b"\x11" * 16), ("v.patch", b"\x22" * 8), ("output.w", b"\x33" * 12)]
    kvs = [
        _kv_string("general.architecture", "toy"),
        _kv_u32("toy.block_count", 1),
        _kv_u32("toy.vision.patch_size", 16),
        struct.pack("<Q", 7)
        + b"toy.arr"
        + struct.pack("<IIQ", 9, 4, 3)
        + struct.pack("<3I", 1, 2, 3),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    offset = 0
    for name, data in tensors:
        header += struct.pack("<Q", len(name)) + name.encode()
        header += struct.pack("<IQ", 1, len(data) // 4) + struct.pack("<IQ", 0, offset)
        offset += -(-len(data) // 32) * 32
    body = header + b"\0" * (-len(header) % 32)
    for _, data in tensors:
        body += data + b"\0" * (-len(data) % 32)
    path.write_bytes(body)


def test_strip_vision_drops_only_vision_tensors_and_keys(tmp_path):
    source, target = tmp_path / "toy.gguf", tmp_path / "toy-text.gguf"
    _write_toy_gguf(source)
    assert text_only.strip_vision(source, target) == (2, 1)
    data = target.read_bytes()
    assert b"toy.vision" not in data and b"v.patch" not in data
    assert b"\x11" * 16 in data and b"\x33" * 12 in data and b"\x22" * 8 not in data
    assert b"toy.block_count" in data and b"toy.arr" in data
    # the copy is a valid GGUF again: stripping it once more keeps both tensors
    assert text_only.strip_vision(target, tmp_path / "again.gguf") == (2, 0)
