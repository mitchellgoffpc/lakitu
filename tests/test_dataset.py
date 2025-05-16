import av
import numpy as np
import unittest
from pathlib import Path

from lakitu.datasets.dataset import EpisodeDataset
from lakitu.datasets.format import load_data

class TestEpisodeDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent.parent / "lakitu/data/episodes"
        self.assertTrue(self.data_dir.exists(), f"Data directory {self.data_dir} does not exist")

        episode_dirs = sorted(self.data_dir.iterdir())
        self.assertTrue(len(episode_dirs) > 0, "No episodes found in data directory")
        self.episode_dir = episode_dirs[0]

        container = av.open(self.episode_dir / "episode.mp4")
        self.frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        self.actions = load_data(self.episode_dir / "episode.data")['action.buttons']

    def create_dataset(self, frames_per_sample=1):
        deltas = list(range(frames_per_sample))
        dataset = EpisodeDataset(data_dir=self.data_dir, deltas={'observation.image': deltas, 'action.buttons': deltas})
        episode_name = self.episode_dir.name
        dataset.episodes = {episode_name: dataset.episodes[episode_name]}
        dataset.episodes_by_idx = [episode_name] * len(dataset.episodes[episode_name].data)
        return dataset

    def test_dataset_length(self):
        """Test that the dataset length is correct."""
        dataset = self.create_dataset()
        expected_length = len(self.frames)
        self.assertEqual(len(dataset), expected_length, f"Dataset length {len(dataset)} doesn't match expected {expected_length}")

    def test_single_frame_retrieval(self):
        """Test retrieving single frame samples."""
        dataset = self.create_dataset()

        for idx in range(0, len(self.frames) - 1, 23):
            batch = dataset[idx]
            frames = batch['observation.image']
            actions = batch['action.buttons']
            self.assertEqual(frames.shape, (1, 3, 240, 320), f"Frame shape mismatch at index {idx}")
            self.assertEqual(actions.shape, (1, 14), f"Action shape mismatch at index {idx}")

            frame_np = (frames[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            self.assertTrue(np.all(frame_np == self.frames[idx]), f"Frame content mismatch at index {idx}")
            self.assertTrue(np.all(actions[0].numpy() == self.actions[idx]), f"Action content mismatch at index {idx}")

    def test_multi_frame_retrieval(self):
        """Test retrieving multi-frame samples."""
        frames_per_sample = 4
        dataset = self.create_dataset(frames_per_sample=frames_per_sample)

        for idx in range(0, min(len(dataset), len(self.frames) - frames_per_sample + 1), 23):
            batch = dataset[idx]
            frames = batch['observation.image']
            actions = batch['action.buttons']
            self.assertEqual(frames.shape, (frames_per_sample, 3, 240, 320), f"Frames shape mismatch at index {idx}")
            self.assertEqual(actions.shape, (frames_per_sample, 14), f"Actions shape mismatch at index {idx}")

            for i in range(frames_per_sample):
                frame_np = (frames[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                self.assertTrue(np.all(frame_np == self.frames[idx + i]), f"Frame content mismatch at index {idx+i}")
                self.assertTrue(np.all(actions[i].numpy() == np.array(self.actions[idx + i])), f"Action content mismatch at index {idx+i}")
