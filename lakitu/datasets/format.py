import json
import struct
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO

Field = tuple[str, np.dtype, tuple[int, ...]]

# Normally we'd use parquet or .npy files for our tabular data, but since we'll be streaming the data to disk
# as we record, it's convenient to use a format that's more append-friendly. The format we use is similar in nature
# to the .npy or .safetensors formats, with a few small modifications. Its binary representation is as follows:
# - 4 bytes: N, the length of the header
# - N bytes: A JSON UTF-8 string representing the header. The header will be a JSON array of objects, each with the following fields:
#   - name: The name of the field
#   - dtype: The data type of the field, consumable by np.dtype() (e.g. 'float32', 'int32', etc.)
#   - shape: The shape of the field. This will be a JSON array of integers, or an empty array if the field is scalar. It excludes the
#            first dimension, which will be inferred from the data. This allows us to append rows without modifying the header.
# - Rest of file: Byte buffer

def load_data(f: Path | BinaryIO) -> np.ndarray:
    with open(f, 'rb') if isinstance(f, Path) else nullcontext(f) as f:
        header_length = struct.unpack('<I', f.read(4))[0]
        fields = json.loads(f.read(header_length).decode('utf-8'))
        buffer = f.read()
        row_size = sum(np.dtype(field['dtype']).itemsize * (np.prod(field['shape']) if field['shape'] else 1) for field in fields)
        assert len(buffer) % row_size == 0, f"Data size {len(buffer)} is not a multiple of row size {row_size}"
        return np.frombuffer(buffer, dtype=[(field['name'], np.dtype(field['dtype']), tuple(field['shape'])) for field in fields])

class Writer:
    def __init__(self, f: BinaryIO, fields: list[Field]):
        self.output_stream = f
        self.fields = {name: (dtype, shape) for name, dtype, shape in fields}
        self.row_size = sum(np.dtype(dtype).itemsize * (np.prod(shape) if shape else 1) for _, dtype, shape in fields)
        header_dict = [{'name': name, 'dtype': np.dtype(dtype).name, 'shape': shape} for name, dtype, shape in fields]
        header_bytes = json.dumps(header_dict).encode('utf-8')
        header_bytes = struct.pack('<I', len(header_bytes)) + header_bytes
        self.output_stream.write(header_bytes)

    def writerow(self, row: dict[str, np.ndarray]) -> None:
        assert row.keys() == self.fields.keys(), f"Row keys {list(row.keys())} do not match expected columns {list(self.fields.keys())}"
        for name, (dtype, shape) in self.fields.items():
            assert row[name].dtype == dtype, f"Row field {name} has dtype {row[name].dtype}, expected {dtype}"
            assert row[name].shape == shape, f"Row field {name} has shape {row[name].shape}, expected {shape}"
        row_bytes = b''.join(row[name].tobytes() for name in self.fields.keys())
        assert len(row_bytes) == self.row_size, f"Row size mismatch: {len(row_bytes)} != {self.row_size}"
        self.output_stream.write(row_bytes)
