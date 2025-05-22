from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset

from lakitu.datasets.format import load_data
from lakitu.datasets.vidindex import CodecType, get_frame_info, get_mp4_boxes, get_decode_range

DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'episodes'
VIDEO_KEY = 'observation.image'

@dataclass
class EpisodeData:
    name: str
    path: Path
    data: np.ndarray
    codec: CodecType
    frame_info: list[tuple[int, int, bool, int]]
    extradata: bytes
    start_idx: int
    end_idx: int

class EpisodeDataset(Dataset):
    def __init__(self, data_dirs: list[Path], deltas: dict[str, list[int]]) -> None:
        self.data_dirs = data_dirs
        self.deltas = deltas or {}
        self.episodes = {}
        self.episodes_by_idx: list[str] = []

        for data_dir in self.data_dirs:
            for episode_dir in sorted(data_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                episode_name = episode_dir.name
                data_file = episode_dir / "episode.data"
                video_file = episode_dir / "episode.mp4"
                assert data_file.exists(), f"Missing {data_file}"
                assert video_file.exists(), f"Missing {video_file}"

                data = load_data(data_file)
                boxes = get_mp4_boxes(video_file)
                codec_type, frame_info, extradata = get_frame_info(boxes)
                start_idx = len(self.episodes_by_idx)
                end_idx = start_idx + len(data)
                assert len(frame_info) == len(data), f"Mismatch between video and data length for {video_file.name}"
                self.episodes[episode_name] = \
                    EpisodeData(episode_name, data_dir / episode_name, data, codec_type, frame_info, extradata, start_idx, end_idx)
                self.episodes_by_idx.extend([episode_name] * len(data))

        # Need to create the decoders lazily since they can't be pickled for the multiprocessing workers
        self.decoders: dict[CodecType, av.VideoCodecContext] = {}
        self.decoder_ages: dict[CodecType, int] = {}

    def __len__(self) -> int:
        return len(self.episodes_by_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

    def get_frame_data(self, episode: EpisodeData, frame_idx: int, frame_deltas: list[int]) -> tuple[np.ndarray, int, int]:
        decode_start_idx, decode_end_idx = \
            get_decode_range(episode.frame_info, frame_idx + frame_deltas[0], frame_deltas[-1] - frame_deltas[0] + 1)

        # For some reason, the decoder seems to get slower over time.
        # We can remedy this by recreating the decoder every so often,
        # but we should really try to understand why this occurs.
        if episode.codec not in self.decoders or self.decoder_ages[episode.codec] >= 100:
            self.decoder_ages[episode.codec] = 0
            self.decoders[episode.codec] = av.CodecContext.create(episode.codec.value, 'r')  # type: ignore
        self.decoder_ages[episode.codec] += 1

        boxes = get_mp4_boxes(episode.path / "episode.mp4")
        video_data = boxes['mdat']
        decoder: av.VideoCodecContext = self.decoders[episode.codec]
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

    def get_tabular_data(self, episode: EpisodeData, frame_idx: int, deltas: list[int], key: str) -> tuple[np.ndarray, int, int]:
        start_idx = max(frame_idx + deltas[0], 0)
        end_idx = min(frame_idx + deltas[-1] + 1, len(episode.data))
        data = list(episode.data[start_idx:end_idx][key])
        return data, start_idx, end_idx


# Entry point for testing

if __name__ == "__main__":
    import time
    import argparse
    from tqdm import tqdm
    from lakitu.env.gym import draw_actions, draw_info

    parser = argparse.ArgumentParser(description='Dataset visualization and benchmarking')
    parser.add_argument('mode', choices=['benchmark', 'visualize'], help='Mode to run in')
    parser.add_argument('-d', '--data-dir', type=str, default=str(DEFAULT_DATA_DIR), help='Path to the data directory')
    parser.add_argument('-f', '--frames-per-sample', type=int, default=None, help='Number of frames per sample')
    args = parser.parse_args()

    delta_range = list(range(-args.frames_per_sample + 1, 1)) if args.frames_per_sample else [0]
    deltas = {key: delta_range for key in ['observation.image', 'action.joystick', 'action.buttons', 'info.level']}
    dataset = EpisodeDataset([Path(args.data_dir)], deltas=deltas)

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
        screen = pygame.display.set_mode((W * 2, H * 2 + 100))
        clock = pygame.time.Clock()

        running = True
        frames: list[np.ndarray] = []
        frame_idx = 0
        batch = {}
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif not args.frames_per_sample and event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    next_episode = next((ep for ep in dataset.episodes.values() if ep.start_idx > frame_idx), None)
                    frame_idx = next_episode.start_idx if next_episode else 0
                    frames = []

            if not frames:
                batch = dataset[np.random.randint(len(dataset))] if args.frames_per_sample else dataset[frame_idx]
                frames_tensor = batch['observation.image']
                frames = list((frames_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))

            # Draw frame
            frame_idx += 1
            frame = frames.pop(0)
            frame = cv2.resize(frame, (W * 2, H * 2))
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

                        # Draw actions
            current_idx = len(batch['action.joystick']) - len(frames) - 1
            info = {k.removeprefix('info.'): batch[k][current_idx] for k in batch if k.startswith('info.') and not k.endswith('.padded')}
            draw_actions(screen, batch['action.joystick'][current_idx], batch['action.buttons'][current_idx], H * 2, W * 2, 100)
            draw_info(screen, info, H * 2, W * 2)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
