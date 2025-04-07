import av
import csv
import unittest
import numpy as np
from pathlib import Path
from lakitu.datasets.dataset import RolloutDataset

class TestRolloutDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent.parent / "lakitu/data/rollouts"
        self.assertTrue(self.data_dir.exists(), f"Data directory {self.data_dir} does not exist")

        rollout_dirs = sorted(self.data_dir.iterdir())
        self.assertTrue(len(rollout_dirs) > 0, "No rollouts found in data directory")
        self.rollout_dir = rollout_dirs[0]

        container = av.open(self.rollout_dir / "observations.mp4")
        self.frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]

        self.actions = []
        with open(self.rollout_dir / 'actions.csv', 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if int(row['controller_index']) == 0:
                    self.actions.append([float(row[key]) for key in fieldnames or []])

    def test_dataset_length(self):
        """Test that the dataset length is correct."""
        dataset = RolloutDataset(data_dir=self.data_dir, frames_per_sample=4, rollout_ids=[self.rollout_dir.name])
        expected_length = len(self.frames) - dataset.frames_per_sample + 1
        self.assertEqual(len(dataset), expected_length, f"Dataset length {len(dataset)} doesn't match expected {expected_length}")

    def test_single_frame_retrieval(self):
        """Test retrieving single frame samples."""
        single_frame_dataset = RolloutDataset(data_dir=self.data_dir, frames_per_sample=1, rollout_ids=[self.rollout_dir.name])

        for idx in range(0, len(self.frames) - 1, 23):
            frames, actions = single_frame_dataset[idx]
            self.assertEqual(frames.shape, (1, 3, 240, 320), f"Frame shape mismatch at index {idx}")
            self.assertEqual(actions.shape, (1, len(self.actions[0])), f"Action shape mismatch at index {idx}")

            frame_np = (frames[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            frame_diff = np.abs(frame_np.astype(float) - self.frames[idx].astype(float)).mean()
            self.assertLess(frame_diff, 1.0, f"Frame content mismatch at index {idx}")

            action_diff = np.abs(actions[0].numpy() - np.array(self.actions[idx])).mean()
            self.assertLess(action_diff, 1e-5, f"Action content mismatch at index {idx}")

    def test_multi_frame_retrieval(self):
        """Test retrieving multi-frame samples."""
        dataset = RolloutDataset(data_dir=self.data_dir, frames_per_sample=4, rollout_ids=[self.rollout_dir.name])

        for idx in range(0, min(len(dataset), len(self.frames) - dataset.frames_per_sample + 1), 23):
            frames, actions = dataset[idx]
            self.assertEqual(frames.shape, (dataset.frames_per_sample, 3, 240, 320), f"Frames shape mismatch at index {idx}")
            self.assertEqual(actions.shape, (dataset.frames_per_sample, len(self.actions[0])), f"Actions shape mismatch at index {idx}")

            for i in range(dataset.frames_per_sample):
                frame_np = (frames[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                frame_diff = np.abs(frame_np.astype(float) - self.frames[idx + i].astype(float)).mean()
                self.assertLess(frame_diff, 1.0, f"Frame content mismatch at index {idx+i}")

                action_diff = np.abs(actions[i].numpy() - np.array(self.actions[idx + i])).mean()
                self.assertLess(action_diff, 1e-5, f"Action content mismatch at index {idx+i}")
