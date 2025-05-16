import io
import unittest
import numpy as np

from lakitu.datasets.format import Field, Writer, load_data

class TestDatasetFormat(unittest.TestCase):
    def test_basic_dataset(self):
        # Create test data
        fields = [
            ('position', np.dtype(np.float32), (3,)),
            ('velocity', np.dtype(np.float32), (3,)),
            ('score', np.dtype(np.float32), ()),
            ('lives', np.dtype(np.int32), ()),
        ]

        data = {
            'position': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'velocity': np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'score': np.array(100.0, dtype=np.float32),
            'lives': np.array(3, dtype=np.int32),
        }

        # Write data
        buffer = io.BytesIO()
        writer = Writer(buffer, fields)
        writer.writerow(data)
        writer.writerow(data)  # Write twice to test multiple rows

        # Read data back
        buffer.seek(0)
        result = load_data(buffer)

        # Verify results
        self.assertEqual(len(result), 2)
        for i in range(2):
            np.testing.assert_array_equal(result['position'][i], data['position'])
            np.testing.assert_array_equal(result['velocity'][i], data['velocity'])
            np.testing.assert_array_equal(result['score'][i], data['score'])
            np.testing.assert_array_equal(result['lives'][i], data['lives'])

    def test_missing_key(self):
        fields: list[Field] = [('x', np.dtype(np.float32), ()), ('y', np.dtype(np.float32), ())]
        data = {'x': np.array(1.0, dtype=np.float32)}  # y is missing

        buffer = io.BytesIO()
        writer = Writer(buffer, fields)

        with self.assertRaises(AssertionError) as context:
            writer.writerow(data)
        self.assertEqual(str(context.exception), "Row keys ['x'] do not match expected columns ['x', 'y']")

    def test_extra_key(self):
        fields: list[Field] = [('x', np.dtype(np.float32), ())]
        data = {
            'x': np.array(1.0, dtype=np.float32),
            'y': np.array(2.0, dtype=np.float32),  # Extra field
        }

        buffer = io.BytesIO()
        writer = Writer(buffer, fields)

        with self.assertRaises(AssertionError) as context:
            writer.writerow(data)
        self.assertEqual(str(context.exception), "Row keys ['x', 'y'] do not match expected columns ['x']")

    def test_wrong_dtype(self):
        fields: list[Field] = [('x', np.dtype(np.float32), ())]
        data = {'x': np.array(1, dtype=np.int32)}  # Wrong dtype

        buffer = io.BytesIO()
        writer = Writer(buffer, fields)

        with self.assertRaises(AssertionError) as context:
            writer.writerow(data)
        self.assertEqual(str(context.exception), "Row field x has dtype int32, expected float32")

    def test_wrong_shape(self):
        fields: list[Field] = [('x', np.dtype(np.float32), (2,))]
        data = {'x': np.array([1.0, 2.0, 3.0], dtype=np.float32)}  # Wrong shape

        buffer = io.BytesIO()
        writer = Writer(buffer, fields)

        with self.assertRaises(AssertionError) as context:
            writer.writerow(data)
        self.assertEqual(str(context.exception), "Row field x has shape (3,), expected (2,)")

    def test_corrupted_data(self):
        fields: list[Field] = [('x', np.dtype(np.float32), ())]
        data = {'x': np.array(1.0, dtype=np.float32)}

        buffer = io.BytesIO()
        writer = Writer(buffer, fields)
        writer.writerow(data)

        # Append an incomplete row
        buffer.write(b'incomplete')
        buffer.seek(0)

        with self.assertRaises(AssertionError) as context:
            load_data(buffer)
        self.assertEqual(str(context.exception), "Data size 14 is not a multiple of row size 4")


if __name__ == '__main__':
    unittest.main()
