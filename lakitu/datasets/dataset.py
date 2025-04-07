import av
import csv
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from lakitu.datasets.vidindex import get_h264_index, get_mp4_boxes, get_decode_range

class RolloutDataset(Dataset):
    def __init__(self, data_dir="lakitu/data/rollouts", frames_per_sample=1, rollout_ids=None, transform=None):
        self.data_dir = Path(data_dir)
        self.frames_per_sample = frames_per_sample
        self.transform = transform
        self.rollouts = []
        self.rollouts_by_idx: list[int] = []
        self.codec = av.CodecContext.create('h264', 'r')

        # Create rollout index
        for rollout_idx, rollout_dir in enumerate(sorted(self.data_dir.iterdir())):
            if rollout_ids is not None and rollout_dir.name not in rollout_ids:
                continue
            info_path = rollout_dir / "info.json"
            video_path = rollout_dir / "observations.mp4"
            actions_path = rollout_dir / "actions.csv"

            if not rollout_dir.is_dir():
                continue
            if not (info_path.exists() and video_path.exists() and actions_path.exists()):
                continue

            with open(info_path, 'r') as f:
                info = json.load(f)
            num_steps = info['num_steps']
            if num_steps <= 0:
                continue

            num_samples = num_steps - frames_per_sample + 1
            self.rollouts.append({
                'path': rollout_dir,
                'start_idx': len(self.rollouts_by_idx),
                'end_idx': len(self.rollouts_by_idx) + num_samples,
            })
            self.rollouts_by_idx.extend([rollout_idx] * num_samples)

    def __len__(self):
        return len(self.rollouts_by_idx)

    def __getitem__(self, idx):
        rollout_idx = self.rollouts_by_idx[idx]
        rollout = self.rollouts[rollout_idx]
        frame_idx = idx - rollout['start_idx']

        boxes = get_mp4_boxes(rollout['path'] / 'observations.mp4')
        frame_types, extradata = get_h264_index(boxes)
        h264_data = boxes['mdat']
        self.codec.extradata = extradata

        # Decode the packets
        frames = []
        start_idx, end_idx = get_decode_range(frame_types, frame_idx, self.frames_per_sample)
        for i in range(start_idx, end_idx):
            _, slice_type, offset = frame_types[i]
            next_offset = frame_types[i+1][2] if i < len(frame_types) - 1 else len(h264_data)

            packet = av.Packet(h264_data[offset:next_offset])
            packet.is_keyframe = slice_type in (2, 7)
            packet.pts = i
            packet.dts = i

            for frame in self.codec.decode(packet):
                if frame.pts is None:
                    raise ValueError(f"Failed to decode frame {i} at {rollout['path'].name}/{frame_idx}")
                if frame_idx <= i < frame_idx + self.frames_per_sample:
                    frames.append(frame.to_ndarray(format='rgb24'))

        # Load actions
        actions = []
        with open(rollout['path'] / 'actions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['controller_index']) == 0 and frame_idx <= int(row['frame_index']) < frame_idx + self.frames_per_sample:
                    actions.append([float(row[key]) for key in reader.fieldnames or []])

        if len(actions) != self.frames_per_sample:
            raise ValueError(f"Expected {self.frames_per_sample} actions at {rollout['path'].name}/{frame_idx}, but got {len(actions)}")

        # Create result tensors and transform
        frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
        actions_tensor = torch.tensor(actions).float()
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        return frames_tensor, actions_tensor


if __name__ == "__main__":
    import time
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Dataset visualization and benchmarking')
    parser.add_argument('mode', choices=['benchmark', 'visualize'], help='Mode to run in')
    parser.add_argument('-f', '--frames-per-sample', type=int, default=4, help='Number of frames per sample')
    args = parser.parse_args()

    dataset = RolloutDataset(frames_per_sample=args.frames_per_sample)

    if args.mode == 'benchmark':
        dataset[0]
        num_samples = 50
        start_time = time.time()
        for idx in tqdm(np.random.randint(0, len(dataset), size=num_samples)):
            dataset[idx]
        elapsed = time.time() - start_time
        print(f"Processed {num_samples} samples in {elapsed:.2f} seconds")
        print(f"Average speed: {num_samples/elapsed:.2f} samples/second")

    elif args.mode == 'visualize':
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((320, 240))
        clock = pygame.time.Clock()

        running = True
        frames: list[np.ndarray] = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not frames:
                frames_tensor, _ = dataset[np.random.randint(len(dataset))]
                frames = list((frames_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))

            frame = frames.pop(0)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
