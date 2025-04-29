import av
import json
import torch
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset

from lakitu.datasets.vidindex import CodecType, get_frame_info, get_mp4_boxes, get_decode_range

DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'episodes'
VIDEO_KEY = 'observation.image'


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

def load_episode_data(data_file):
    with open(data_file, 'rb') as f:
        header_length = struct.unpack('<I', f.read(4))[0]
        fields = json.loads(f.read(header_length).decode('utf-8'))
        buffer = f.read()
        row_size = sum(np.dtype(field['dtype']).itemsize * (np.prod(field['shape']) if field['shape'] else 1) for field in fields)
        assert len(buffer) % row_size == 0, f"Data size {len(buffer)} is not a multiple of row size {row_size}"
        return np.frombuffer(buffer, dtype=[(field['name'], np.dtype(field['dtype']), tuple(field['shape'])) for field in fields])


# Episode Dataset

@dataclass
class EpisodeData:
    name: str
    data: np.ndarray
    codec: CodecType
    frame_info: list[tuple[int, int, bool, int]]
    extradata: bytes
    start_idx: int
    end_idx: int

class EpisodeDataset(Dataset):
    def __init__(self, data_dir=DEFAULT_DATA_DIR, deltas=None):
        self.data_dir = Path(data_dir)
        self.deltas = deltas or {}
        self.episodes = {}
        self.episodes_by_idx: list[str] = []

        for episode_dir in sorted(self.data_dir.iterdir()):
            episode_name = episode_dir.name
            data_file = episode_dir / "episode.data"
            video_file = episode_dir / "episode.mp4"
            assert data_file.exists(), f"Missing {data_file}"
            assert video_file.exists(), f"Missing {video_file}"

            data = load_episode_data(data_file)
            boxes = get_mp4_boxes(video_file)
            codec_type, frame_info, extradata = get_frame_info(boxes)
            start_idx = len(self.episodes_by_idx)
            end_idx = start_idx + len(data)
            assert len(frame_info) == len(data), f"Mismatch between video and data length for {video_file.name}"
            self.episodes[episode_name] = EpisodeData(episode_name, data, codec_type, frame_info, extradata, start_idx, end_idx)
            self.episodes_by_idx.extend([episode_name] * len(data))

        codecs = set(ep.codec for ep in self.episodes.values())
        self.decoders: dict[CodecType, av.VideoCodecContext | None] = {codec: None for codec in codecs}
        self.decoder_ages = {codec: 0 for codec in codecs}

    def __len__(self):
        return len(self.episodes_by_idx)

    def __getitem__(self, idx):
        episode_name = self.episodes_by_idx[idx]
        episode = self.episodes[episode_name]
        frame_idx = idx - episode.start_idx

        batch = {}
        for key, deltas in self.deltas.items():
            if key == VIDEO_KEY:
                data, start_idx, end_idx = self.get_frame_data(episode, frame_idx, deltas)
            else:
                data, start_idx, end_idx = self.get_tabular_data(episode, frame_idx, deltas, key)

            start_pad_len = max(0, start_idx - (frame_idx + deltas[0]))
            end_pad_len = max(0, (frame_idx + deltas[-1] + 1) - end_idx)
            padded_data = np.array([data[0]] * start_pad_len + data + [data[-1]] * end_pad_len)
            mask = np.array([1] * start_pad_len + [0] * len(data) + [1] * end_pad_len, dtype=bool)

            expected_num_rows = deltas[-1] - deltas[0] + 1
            assert len(padded_data) == expected_num_rows, \
                f"Expected {expected_num_rows} rows, got {len(padded_data)} at {episode_name}/{key}/{frame_idx}"
            batch[key] = torch.as_tensor(padded_data)
            batch[f'{key}.padded'] = torch.as_tensor(mask)

        if VIDEO_KEY in batch:  # Do this at the end for performance
            batch[VIDEO_KEY] = batch[VIDEO_KEY].permute(0, 3, 1, 2).float() / 255

        return batch

    def get_frame_data(self, episode, frame_idx, frame_deltas):
        decode_start_idx, decode_end_idx = \
            get_decode_range(episode.frame_info, frame_idx + frame_deltas[0], frame_deltas[-1] - frame_deltas[0] + 1)

        # Need to create the decoders lazily since they can't be pickled for the multiprocessing workers
        if self.decoders[episode.codec] is None:
            self.decoders[episode.codec] = av.CodecContext.create(episode.codec.value, 'r')

        # For some reason, the decoder seems to get slower over time.
        # We can remedy this by recreating the decoder every so often,
        # but we should really try to understand why this occurs.
        if self.decoder_ages[episode.codec] >= 100:
            self.decoder_ages[episode.codec] = 0
            self.decoders[episode.codec] = av.CodecContext.create(episode.codec.value, 'r')  # type: ignore
        self.decoder_ages[episode.codec] += 1

        boxes = get_mp4_boxes(self.data_dir / episode.name / "episode.mp4")
        video_data = boxes['mdat']
        decoder = self.decoders[episode.codec]
        decoder.extradata = episode.extradata
        decoder.flush_buffers()

        # Decode the packets
        frames = []
        for i in range(decode_start_idx, decode_end_idx):
            _, _, is_keyframe, offset = episode.frame_info[i]
            next_offset = episode.frame_info[i+1][3] if i < len(episode.frame_info) - 1 else len(video_data)

            packet = av.Packet(video_data[offset:next_offset])
            packet.is_keyframe = is_keyframe
            packet.pts = i
            packet.dts = i

            for frame in decoder.decode(packet):
                frames.append(frame.to_ndarray(format='rgb24'))

        # Flush the decoder
        for frame in decoder.decode(None):
            frames.append(frame.to_ndarray(format='rgb24'))

        start_idx = max(frame_idx + frame_deltas[0], 0)
        end_idx = min(frame_idx + frame_deltas[-1] + 1, len(episode.data))
        frames = frames[start_idx - decode_start_idx:end_idx - decode_start_idx]
        return frames, start_idx, end_idx

    def get_tabular_data(self, episode, frame_idx, deltas, key):
        start_idx = max(frame_idx + deltas[0], 0)
        end_idx = min(frame_idx + deltas[-1] + 1, len(episode.data))
        data = list(episode.data[start_idx:end_idx][key])
        return data, start_idx, end_idx


# Entry point for testing

if __name__ == "__main__":
    import time
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Dataset visualization and benchmarking')
    parser.add_argument('mode', choices=['benchmark', 'visualize'], help='Mode to run in')
    parser.add_argument('-f', '--frames-per-sample', type=int, default=4, help='Number of frames per sample')
    args = parser.parse_args()

    delta_range = list(range(-args.frames_per_sample + 1, 1))
    deltas = {key: delta_range for key in ['observation.image', 'action.joystick', 'action.buttons']}
    dataset = EpisodeDataset(deltas=deltas)

    if args.mode == 'benchmark':
        dataset[0]
        num_samples = 50
        start_time = time.perf_counter()
        for idx in tqdm(np.random.randint(0, len(dataset), size=num_samples)):
            dataset[idx]
        elapsed = time.perf_counter() - start_time
        print(f"Processed {num_samples} samples in {elapsed:.2f} seconds")
        print(f"Average speed: {num_samples/elapsed:.2f} samples/second")

    elif args.mode == 'visualize':
        frame = dataset[0]['observation.image']
        _, _, H, W = frame.shape

        import cv2
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W * 2, H * 2))
        clock = pygame.time.Clock()

        running = True
        frames: list[np.ndarray] = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not frames:
                frames_tensor = dataset[np.random.randint(len(dataset))]['observation.image']
                frames = list((frames_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))

            frame = frames.pop(0)
            frame = cv2.resize(frame, (W * 2, H * 2))
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
