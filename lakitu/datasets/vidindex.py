import struct
from enum import Enum, IntEnum

class CodecType(Enum):
    H264 = "h264"
    AV1 = "libdav1d"

class H264NALUType(IntEnum):
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

class H264SliceType(IntEnum):
    P_SLICE = 0
    B_SLICE = 1
    I_SLICE = 2
    SP_SLICE = 3
    SI_SLICE = 4

class AV1OBUType(IntEnum):
    RESERVED_0 = 0
    SEQUENCE_HEADER = 1
    TEMPORAL_DELIMITER = 2
    FRAME_HEADER = 3
    TILE_GROUP = 4
    METADATA = 5
    FRAME = 6
    REDUNDANT_FRAME_HEADER = 7
    TILE_LIST = 8
    RESERVED_1 = 9
    RESERVED_2 = 10
    RESERVED_3 = 11
    RESERVED_4 = 12
    RESERVED_5 = 13
    RESERVED_6 = 14
    PADDING = 15

class AV1FrameType(IntEnum):
    KEY_FRAME = 0
    INTER_FRAME = 1
    INTRA_ONLY_FRAME = 2
    SWITCH_FRAME = 3

NALU_LENGTH_BYTES = 4
NALU_HEADER_BYTES = 1
OBU_HEADER_BYTES = 1

H264_PARAMETER_SET_NALU_TYPES = (
    H264NALUType.SEQUENCE_PARAMETER_SET,
    H264NALUType.PICTURE_PARAMETER_SET,
)

H264_CODED_SLICE_SEGMENT_NALU_TYPES = (
    H264NALUType.CODED_SLICE_NON_IDR_PICTURE,
    H264NALUType.CODED_SLICE_IDR_PICTURE,
)


# NOTE: The MP4 container format is a bit unhinged, some box types can have custom header data before the subboxes
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
        if box_type in ('moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1', 'av01'):
            offset = {'stsd': 8, 'avc1': 78, 'av01': 78}.get(box_type, 0)  # stsd, avc1, and av01 have header data
            boxes[box_type] = get_mp4_subtree(data[i+offset+8:i+size])
        else:
            boxes[box_type] = data[i+8:i+size]
        i += size
    assert i == len(data), f"Size mismatch: {len(data)} != {i}"
    return boxes

# Credit to https://github.com/commaai/openpilot/blob/master/tools/lib/vidindex.py for the golomb coding logic
def get_ue(data, start_idx, skip_bits):
  prefix_val = 0
  prefix_len = 0
  suffix_val = 0
  suffix_len = 0

  i = start_idx
  while i < len(data):
    j = 7
    while j >= 0:
      if skip_bits > 0:
        skip_bits -= 1
      elif prefix_val == 0:
        prefix_val = (data[i] >> j) & 1
        prefix_len += 1
      else:
        suffix_val = (suffix_val << 1) | ((data[i] >> j) & 1)
        suffix_len += 1
      j -= 1

      if prefix_val == 1 and prefix_len - 1 == suffix_len:
        val = 2**(prefix_len-1) - 1 + suffix_val
        size = prefix_len + suffix_len
        return val, size
    i += 1

def get_leb128(data, start_idx):
    val = 0
    size = 0
    i = start_idx
    while i < len(data):
        byte = data[i]
        val |= (byte & 0x7F) << (size * 7)
        size += 1
        i += 1
        if byte & 0x80 == 0:
            break
    return val, size


# H264 parsing

def get_h264_nalu_len(data, nalu_start_idx):
    assert (data[nalu_start_idx] & 0x80) == 0, "forbidden_zero_bit must be zero"
    assert len(data) > nalu_start_idx+NALU_LENGTH_BYTES, f"NAL unit must be at least {NALU_LENGTH_BYTES} bytes long"
    length = struct.unpack('>I', data[nalu_start_idx:nalu_start_idx+NALU_LENGTH_BYTES])[0]
    assert length > 0, "NAL unit length must be non-zero"
    return length + NALU_LENGTH_BYTES  # AVCC length header doesn't count its own length

def get_h264_nalu_type(data, nalu_start_idx):
    return data[nalu_start_idx+NALU_LENGTH_BYTES] & 0x1F

def get_h264_slice_type(data, nalu_start_idx):
    rbsp_start = nalu_start_idx + NALU_LENGTH_BYTES + NALU_HEADER_BYTES

    is_first_slice = data[rbsp_start] & 0x80 != 0
    if not is_first_slice:
        return (-1, is_first_slice)  # skip non-first slices

    slice_type, _ = get_ue(data, rbsp_start, skip_bits=1)
    return slice_type, is_first_slice

def get_h264_info(data):
    i = 0
    frame_info = []
    while i < len(data):
        nalu_len = get_h264_nalu_len(data, i)
        nalu_type = get_h264_nalu_type(data, i)
        if nalu_type in H264_CODED_SLICE_SEGMENT_NALU_TYPES:
            slice_type, is_first_slice = get_h264_slice_type(data, i)
            is_keyframe = nalu_type == H264NALUType.CODED_SLICE_IDR_PICTURE
            if is_first_slice:
                frame_info.append((nalu_type, slice_type, is_keyframe, i))
        i += nalu_len

    return frame_info


# AV1 parsing

def get_av1_obu_len(data, obu_start_idx):
    assert data[obu_start_idx] & 0x80 == 0, "forbidden_zero_bit must be zero"
    assert data[obu_start_idx] & 0x02 != 0, "obu_has_size_field must be set"
    length, size = get_leb128(data, obu_start_idx+1)
    assert length > 0, "OBU length must be non-zero"
    return length + size + OBU_HEADER_BYTES  # obu_size field doesn't count the header's length

def get_av1_obu_type(data, obu_start_idx):
    return (data[obu_start_idx] & 0x78) >> 3

def get_av1_frame_type(data, obu_start_idx):
    _, size = get_leb128(data, obu_start_idx+1)
    return (data[obu_start_idx+1+size] & 0x60) >> 5

def get_av1_info(data):
    i = 0
    frame_info = []
    prev_obu_type = None
    prev_obu_offset = 0
    while i < len(data):
        obu_len = get_av1_obu_len(data, i)
        obu_type = get_av1_obu_type(data, i)
        if obu_type == AV1OBUType.FRAME:
            frame_type = get_av1_frame_type(data, i)
            is_keyframe = frame_type == AV1FrameType.KEY_FRAME
            offset = prev_obu_offset if prev_obu_type == AV1OBUType.SEQUENCE_HEADER else i
            frame_info.append((obu_type, frame_type, is_keyframe, offset))
        prev_obu_type = obu_type
        prev_obu_offset = i
        i += obu_len

    return frame_info


# Main API functions

def get_frame_info(boxes):
    data = boxes['mdat']
    stsd = boxes['moov']['trak']['mdia']['minf']['stbl']['stsd']  # lol
    if 'avc1' in stsd:
        return CodecType.H264, get_h264_info(data), stsd['avc1']['avcC']
    elif 'av01' in stsd:
        return CodecType.AV1, get_av1_info(data), stsd['av01']['av1C']
    else:
        raise ValueError("No extradata found in mp4 file")

def get_decode_range(frame_info, frame_idx, num_frames):
    start_idx = next((i for i in range(frame_idx, -1, -1) if frame_info[i][2]), 0)
    end_idx = min(frame_idx + num_frames, len(frame_info))
    return start_idx, end_idx


# Entry point for testing

if __name__ == '__main__':
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()

    boxes = get_mp4_boxes(args.input_file)
    codec_type, frame_info, extradata = get_frame_info(boxes)
    print(f"Found {len(frame_info)} frames in index, {len(extradata)} bytes of extradata")

    # Verify that the stream contains only only I frames and P frames
    last_iframe_idx = 0
    gop_sizes = []
    for i, (unit_type, frame_type, _, _) in enumerate(frame_info):
        if codec_type is CodecType.H264 and frame_type not in (H264SliceType.P_SLICE, H264SliceType.I_SLICE):
            raise ValueError(f"Found B-frame or unsupported slice type {frame_type}")
        if codec_type is CodecType.H264 and frame_type == H264SliceType.I_SLICE and unit_type != H264NALUType.CODED_SLICE_IDR_PICTURE:
            raise ValueError(f"Found non-IDR I-frame at index {i}, slice type {frame_type}")
        if codec_type is CodecType.AV1 and frame_type not in (AV1FrameType.KEY_FRAME, AV1FrameType.INTER_FRAME):
            raise ValueError(f"Found unsupported frame type {frame_type}")
        if i > 0 and ((codec_type is CodecType.H264 and frame_type == H264SliceType.I_SLICE) or
                      (codec_type is CodecType.AV1 and frame_type == AV1FrameType.KEY_FRAME)):
            gop_sizes.append(i - last_iframe_idx)
            last_iframe_idx = i

    gop_size_counts = Counter(gop_sizes)
    for gop_size, count in sorted(gop_size_counts.items()):
        print(f"- {count} GOPs of size {gop_size}")
