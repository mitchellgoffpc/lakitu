import av
import torch
import numpy as np
import pyarrow.parquet as pq
from pyarrow import Table
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset

from lakitu.datasets.vidindex import CodecType, get_frame_info, get_mp4_boxes, get_decode_range

DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'episodes'
OBSERVATION_KEY = 'observation'

@dataclass
class EpisodeData:
    data: Table
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
        assert OBSERVATION_KEY in self.deltas, f"deltas must contain key {OBSERVATION_KEY!r}"

        for episode_dir in sorted(self.data_dir.iterdir()):
            episode_name = episode_dir.name
            parquet_file = episode_dir / "episode.parquet"
            video_file = episode_dir / "episode.mp4"
            assert parquet_file.exists(), f"Missing {parquet_file}"
            assert video_file.exists(), f"Missing {video_file}"

            data = pq.read_table(parquet_file)
            boxes = get_mp4_boxes(video_file)
            codec_type, frame_info, extradata = get_frame_info(boxes)
            start_idx = len(self.episodes_by_idx)
            end_idx = start_idx + len(data)
            assert len(frame_info) == len(data), f"Mismatch between video and data length for {video_file.name}"
            self.episodes[episode_name] = EpisodeData(data, codec_type, frame_info, extradata, start_idx, end_idx)
            self.episodes_by_idx.extend([episode_name] * len(data))

        codecs = set(ep.codec for ep in self.episodes.values())
        self.decoders: dict[CodecType, av.VideoCodecContext] = {codec: av.CodecContext.create(codec.value, 'r') for codec in codecs}  # type: ignore
        self.decoder_ages = {codec: 0 for codec in codecs}

    def __len__(self):
        return len(self.episodes_by_idx)

    def __getitem__(self, idx):
        episode_name = self.episodes_by_idx[idx]
        episode = self.episodes[episode_name]
        frame_idx = idx - episode.start_idx
        batch = {}

        # For some reason, the decoder seems to get slower over time.
        # We can remedy this by recreating the decoder every so often,
        # but we should really try to understand why this occurs.
        if self.decoder_ages[episode.codec] >= 100:
            self.decoder_ages[episode.codec] = 0
            self.decoders[episode.codec] = av.CodecContext.create(episode.codec.value, 'r')  # type: ignore
        self.decoder_ages[episode.codec] += 1

        boxes = get_mp4_boxes(self.data_dir / episode_name / "episode.mp4")
        video_data = boxes['mdat']
        decoder = self.decoders[episode.codec]
        decoder.extradata = episode.extradata
        decoder.flush_buffers()

        # Decode the packets
        frames = []
        frame_deltas = self.deltas[OBSERVATION_KEY]
        start_idx, end_idx = get_decode_range(episode.frame_info, frame_idx + frame_deltas[0], frame_deltas[-1] - frame_deltas[0] + 1)
        for i in range(start_idx, end_idx):
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

        # Slice the frames to get the desired range
        frames = frames[max(frame_idx + frame_deltas[0], 0) - start_idx:]
        start_pad_len = max(0, start_idx - (frame_idx + frame_deltas[0]))
        end_pad_len = max(0, (frame_idx + frame_deltas[-1] + 1) - end_idx)
        padded_frames = [frames[0]] * start_pad_len + frames + [frames[-1]] * end_pad_len
        batch[OBSERVATION_KEY] = np.stack(padded_frames).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        batch[f'{OBSERVATION_KEY}.padded'] = np.array([1] * start_pad_len + [0] * len(frames) + [1] * end_pad_len, dtype=bool)

        expected_num_frames = frame_deltas[-1] - frame_deltas[0] + 1
        assert len(padded_frames) == expected_num_frames, \
            f"Expected {expected_num_frames} frames, got {len(padded_frames)} at {episode_name}/observation/{frame_idx}, index {idx}"

        # Slice the pyarrow table to get the desired range
        for key, deltas in self.deltas.items():
            if key == OBSERVATION_KEY:
                continue
            start_idx = max(frame_idx + deltas[0], 0)
            end_idx = min(frame_idx + deltas[-1] + 1, len(episode.data))

            data = episode.data.slice(start_idx, end_idx - start_idx)[key].to_pylist()
            start_pad_len = max(0, start_idx - (frame_idx + deltas[0]))
            end_pad_len = max(0, (frame_idx + deltas[-1] + 1) - end_idx)
            padded_data = [data[0]] * start_pad_len + data + [data[-1]] * end_pad_len
            batch[key] = np.array(padded_data, dtype=np.float32)
            batch[f'{key}.padded'] = np.array([1] * start_pad_len + [0] * len(data) + [1] * end_pad_len, dtype=bool)

            expected_num_rows = deltas[-1] - deltas[0] + 1
            assert len(padded_data) == expected_num_rows, \
                f"Expected {expected_num_rows} rows, got {len(padded_data)} at {episode_name}/{key}/{frame_idx}, index {idx}"

        return {k: torch.as_tensor(v) for k,v in batch.items()}


if __name__ == "__main__":
    import time
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Dataset visualization and benchmarking')
    parser.add_argument('mode', choices=['benchmark', 'visualize'], help='Mode to run in')
    parser.add_argument('-f', '--frames-per-sample', type=int, default=4, help='Number of frames per sample')
    args = parser.parse_args()

    delta_range = list(range(-args.frames_per_sample + 1, 1))
    deltas = {key: delta_range for key in ['observation', 'action']}
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
        frame, _ = dataset[0]
        _, _, H, W = frame.shape

        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        clock = pygame.time.Clock()

        running = True
        frames: list[np.ndarray] = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not frames:
                frames_tensor, _ = dataset[np.random.randint(len(dataset))]
                frames = list(frames_tensor.permute(0, 2, 3, 1).numpy())

            frame = frames.pop(0)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
