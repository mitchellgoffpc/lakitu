import struct
from enum import IntEnum

NAL_UNIT_LENGTH_BYTES = 4
NAL_UNIT_HEADER_BYTES = 1

class H264NalUnitType(IntEnum):
    UNSPECIFIED_0 = 0
    CODED_SLICE_NON_IDR_PICTURE = 1
    CODED_SLICE_DATA_PARTITION_A = 2
    CODED_SLICE_DATA_PARTITION_B = 3
    CODED_SLICE_DATA_PARTITION_C = 4
    CODED_SLICE_IDR_PICTURE = 5
    SUPPLEMENTAL_ENHANCEMENT_INFORMATION = 6
    SEQUENCE_PARAMETER_SET = 7
    PICTURE_PARAMETER_SET = 8
    ACCESS_UNIT_DELIMITER = 9
    END_SEQUENCE = 10
    END_STREAM = 11
    FILLER_DATA = 12
    SEQUENCE_PARAMETER_SET_EXT = 13
    PREFIX_NAL_UNIT = 14
    SUBSET_SEQUENCE_PARAMETER_SET = 15
    RESERVED_0 = 16
    RESERVED_1 = 17
    RESERVED_2 = 18
    CODED_SLICE_AUXILARY_CODED_PICTURE = 19
    CODED_SLICE_EXTENSION = 20
    RESERVED_3 = 21
    RESERVED_4 = 22
    RESERVED_5 = 23
    UNSPECIFIED_1 = 24
    UNSPECIFIED_2 = 25
    UNSPECIFIED_3 = 26
    UNSPECIFIED_4 = 27
    UNSPECIFIED_5 = 28
    UNSPECIFIED_6 = 29
    UNSPECIFIED_7 = 30
    UNSPECIFIED_8 = 31

H264_PARAMETER_SET_NAL_UNIT_TYPES = (
    H264NalUnitType.SEQUENCE_PARAMETER_SET,
    H264NalUnitType.PICTURE_PARAMETER_SET,
)

H264_CODED_SLICE_SEGMENT_NAL_UNITS = (
    H264NalUnitType.CODED_SLICE_NON_IDR_PICTURE,
    H264NalUnitType.CODED_SLICE_IDR_PICTURE,
    H264NalUnitType.CODED_SLICE_DATA_PARTITION_A,
    H264NalUnitType.CODED_SLICE_DATA_PARTITION_B,
    H264NalUnitType.CODED_SLICE_DATA_PARTITION_C,
    H264NalUnitType.CODED_SLICE_AUXILARY_CODED_PICTURE, # ?
    H264NalUnitType.CODED_SLICE_EXTENSION, # ?
)

# NOTE: The MP4 container format is a bit unhinged, ome box types can have custom header data before the subboxes
# so it seems like the only way to parse the full tree is to implement the entire spec a la https://github.com/sannies/mp4parser.
# Fortunately the only boxes we care about are mdat for the the h.264 stream and moov/trak/mdia/minf/stbl/stsd/avc1/avcC
# for the AVCC extradata, so we just extract these two and ignore the rest.

def get_mp4_boxes(input_file):
    with open(input_file, 'rb') as f:
        mp4_data = f.read()
    return get_mp4_subtree(mp4_data)

def get_mp4_subtree(data):
    i = 0
    boxes = {}
    while i < len(data):
        size = struct.unpack('>I', data[i:i+4])[0]
        assert size > 0, "Box size must be greater than 0"
        box_type = data[i+4:i+8].decode()
        if box_type in ('moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1'):
            offset = {'stsd': 8, 'avc1': 78}.get(box_type, 0)  # stsd and avc1 have header data
            boxes[box_type] = get_mp4_subtree(data[i+offset+8:i+size])
        else:
            boxes[box_type] = data[i+8:i+size]
        i += size
    assert i == len(data), f"Size mismatch: {len(data)} != {i}"
    return boxes

# Credit to https://github.com/commaai/openpilot/blob/master/tools/lib/vidindex.py for the golomb coding logic
def get_ue(dat, start_idx, skip_bits):
  prefix_val = 0
  prefix_len = 0
  suffix_val = 0
  suffix_len = 0

  i = start_idx
  while i < len(dat):
    j = 7
    while j >= 0:
      if skip_bits > 0:
        skip_bits -= 1
      elif prefix_val == 0:
        prefix_val = (dat[i] >> j) & 1
        prefix_len += 1
      else:
        suffix_val = (suffix_val << 1) | ((dat[i] >> j) & 1)
        suffix_len += 1
      j -= 1

      if prefix_val == 1 and prefix_len - 1 == suffix_len:
        val = 2**(prefix_len-1) - 1 + suffix_val
        size = prefix_len + suffix_len
        return val, size
    i += 1

def get_h264_nal_unit_len(data, nal_unit_start):
    assert (data[nal_unit_start] & 0x80) == 0, "forbidden_zero_bit must be zero"
    assert len(data) > nal_unit_start+NAL_UNIT_LENGTH_BYTES, f"NAL unit must be at least {NAL_UNIT_LENGTH_BYTES} bytes long"
    length = struct.unpack('>I', data[nal_unit_start:nal_unit_start+NAL_UNIT_LENGTH_BYTES])[0]
    assert length > 0, "NAL unit length must be non-zero"
    return length + NAL_UNIT_LENGTH_BYTES  # AVCC length header doesn't count its own length

def get_h264_nal_unit_type(data, nal_unit_start):
    return data[nal_unit_start+NAL_UNIT_LENGTH_BYTES] & 0x1F

def get_h264_slice_type(data, nal_unit_start):
    rbsp_start = nal_unit_start + NAL_UNIT_LENGTH_BYTES + NAL_UNIT_HEADER_BYTES

    is_first_slice = data[rbsp_start] & 0x80 != 0
    if not is_first_slice:
        return (-1, is_first_slice)  # skip non-first slices

    slice_type, _ = get_ue(data, rbsp_start, skip_bits=1)
    return slice_type, is_first_slice

def get_h264_index(boxes):
    data = boxes['mdat']
    extradata = boxes['moov']['trak']['mdia']['minf']['stbl']['stsd']['avc1']['avcC']  # lol

    i = 0
    frame_types = []
    while i < len(data):
        nal_unit_len = get_h264_nal_unit_len(data, i)
        nal_unit_type = get_h264_nal_unit_type(data, i)
        if nal_unit_type in H264_CODED_SLICE_SEGMENT_NAL_UNITS:
            slice_type, is_first_slice = get_h264_slice_type(data, i)
            if is_first_slice:
                frame_types.append((nal_unit_type, slice_type, i))
        i += nal_unit_len

    return frame_types, extradata

def get_decode_range(frame_types, frame_idx, num_frames):
    start_idx = next(i for i in range(frame_idx, -1, -1) if frame_types[i][0] == H264NalUnitType.CODED_SLICE_IDR_PICTURE)
    end_idx = next(i for i in range(frame_idx + num_frames, len(frame_types)) if frame_types[i][1] in (2, 7))
    return start_idx, end_idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()

    boxes = get_mp4_boxes(args.input_file)
    frame_types, extradata = get_h264_index(boxes)
    print(f"Found {len(frame_types)} frames in index, {len(extradata)} bytes of extradata")

    # Verify that the stream contains only only I frames and P frames
    last_iframe_idx = 0
    gop_sizes = []
    for i, (nal_unit_type, slice_type, _) in enumerate(frame_types):
        if slice_type not in (0, 5, 2, 7):  # P-slice = 0/5, I-slice = 2/7
            raise ValueError(f"Found B-frame or unexpected slice type {slice_type}")
        if slice_type in (2, 7) and nal_unit_type != H264NalUnitType.CODED_SLICE_IDR_PICTURE:
            raise ValueError(f"Found non-IDR I-frame at index {i}, slice type {slice_type}")
        if slice_type in (2, 7) and i > 0:
            gop_sizes.append(i - last_iframe_idx)
            last_iframe_idx = i

    nonstandard_gop_sizes = [x for x in gop_sizes if x != 30]
    print(f"Found {len(nonstandard_gop_sizes)} non-standard GOP sizes: {nonstandard_gop_sizes}")
