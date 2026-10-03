"""Register a text-only copy of an Ollama model: same weights, chat template and defaults, without
the vision tower (docs 16.4a).

    PYTHONPATH=. .venv/bin/python scripts/ollama_text_only.py --source qwen3.5:4b --name qwen3.5:4b-text

Ollama's qwen3.5 build bundles the vision encoder in the model file and loads it onto the GPU
(about 740 MiB for the 4B at a 32k context) although Vestige sends text only. The copy drops the
``v.*`` tensors and ``<arch>.vision.*`` keys; every other tensor is copied byte for byte, and the
Modelfile (renderer, parser, parameters, licence) is reused unchanged apart from ``FROM``.
Measured 2026-10-02: identical verifier answers to qwen3.5:4b, GPU memory 4547 -> 3807 MiB.
Honours ``OLLAMA_HOST`` like the ``ollama`` CLI; ``--ollama`` picks the binary.
"""

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

T_STRING, T_ARRAY = 8, 9
SCALAR = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]


def _read_value(handle: BinaryIO, vtype: int) -> Any:
    if vtype == T_STRING:
        return handle.read(_read(handle, "<Q"))
    if vtype == T_ARRAY:
        atype = _read(handle, "<I")
        return atype, [_read_value(handle, atype) for _ in range(_read(handle, "<Q"))]
    return _read(handle, SCALAR[vtype])


def _write_value(out: BinaryIO, vtype: int, value: Any) -> None:
    if vtype == T_STRING:
        out.write(struct.pack("<Q", len(value)) + value)
    elif vtype == T_ARRAY:
        atype, items = value
        out.write(struct.pack("<IQ", atype, len(items)))
        for item in items:
            _write_value(out, atype, item)
    else:
        out.write(struct.pack(SCALAR[vtype], value))


def strip_vision(source: Path, target: Path) -> tuple[int, int]:
    """Copy GGUF ``source`` to ``target`` without vision tensors/keys; returns (kept, dropped)."""
    with source.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{source} is not a GGUF file")
        version = _read(handle, "<I")
        n_tensors, n_kv = _read(handle, "<Q"), _read(handle, "<Q")
        kvs = []
        for _ in range(n_kv):
            key = handle.read(_read(handle, "<Q")).decode()
            vtype = _read(handle, "<I")
            kvs.append((key, vtype, _read_value(handle, vtype)))
        tensors = []
        for _ in range(n_tensors):
            name = handle.read(_read(handle, "<Q")).decode()
            dims = [_read(handle, "<Q") for _ in range(_read(handle, "<I"))]
            tensors.append([name, dims, _read(handle, "<I"), _read(handle, "<Q")])
        align = next((v for k, _, v in kvs if k == "general.alignment"), 32)
        data_start = -(-handle.tell() // align) * align
        data_size = source.stat().st_size - data_start
        offsets = sorted(t[3] for t in tensors) + [data_size]
        for tensor in tensors:  # bytes up to the next tensor (includes alignment padding)
            tensor.append(offsets[offsets.index(tensor[3]) + 1] - tensor[3])
        arch = next(v for k, _, v in kvs if k == "general.architecture").decode()
        kvs = [kv for kv in kvs if not kv[0].startswith(f"{arch}.vision.")]
        kept = [t for t in tensors if not t[0].startswith("v.")]
        with target.open("wb") as out:
            out.write(b"GGUF" + struct.pack("<IQQ", version, len(kept), len(kvs)))
            for key, vtype, value in kvs:
                out.write(struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", vtype))
                _write_value(out, vtype, value)
            offset = 0
            for name, dims, ttype, _, size in kept:
                out.write(struct.pack("<Q", len(name)) + name.encode())
                out.write(
                    struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                )
                out.write(struct.pack("<IQ", ttype, offset))
                offset += -(-size // align) * align
            out.write(b"\0" * (-out.tell() % align))
            for _, _, _, old_offset, size in kept:
                handle.seek(data_start + old_offset)
                remaining = size
                while remaining:
                    chunk = handle.read(min(remaining, 64 << 20))
                    out.write(chunk)
                    remaining -= len(chunk)
                out.write(b"\0" * (-size % align))
    return len(kept), len(tensors) - len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="qwen3.5:4b")
    parser.add_argument("--name", default="qwen3.5:4b-text")
    parser.add_argument("--ollama", default="ollama")
    args = parser.parse_args()
    modelfile = subprocess.run(
        [args.ollama, "show", "--modelfile", args.source],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    blob = next(
        (Path(line[5:].strip()) for line in modelfile.splitlines() if line.startswith("FROM /")),
        None,
    )
    if blob is None or not blob.is_file():
        print(f"cannot read the model file of {args.source} (FROM {blob}); pull it first")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "model.gguf"
        kept, dropped = strip_vision(blob, target)
        if not dropped:
            print(f"{args.source} has no vision tensors; nothing to do")
            return 0
        lines = [
            f"FROM {target}" if line.startswith("FROM ") else line
            for line in modelfile.splitlines()
        ]
        (Path(tmp) / "Modelfile").write_text("\n".join(lines) + "\n")
        subprocess.run(
            [args.ollama, "create", args.name, "-f", str(Path(tmp) / "Modelfile")], check=True
        )
    print(f"created {args.name}: kept {kept} tensors, dropped {dropped} vision tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
