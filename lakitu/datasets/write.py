import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np

from lakitu.datasets.format import Field, Writer
from lakitu.env.defs import M64pButtons


def encode(data_queue: mp.Queue, output_path: Path, savestate_path: Optional[str], info_fields: list[Field]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    if savestate_path:
        shutil.copy(savestate_path, output_path / 'initial_state.m64p')

    width, height, fps = 320, 240, 30
    container = av.open(str(output_path / 'episode.mp4'), mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.codec_context.options = {'crf': '23', 'g': '10'}

    data_path = output_path / 'episode.data'
    fields: list[Field] = [
        ('frame_index', np.dtype(np.uint32), ()),
        ('action.joystick', np.dtype(np.float32), (2,)),
        ('action.buttons', np.dtype(np.uint8), (14,)),
        *[(f'info.{name}', dtype, shape) for name, dtype, shape in info_fields],
    ]

    frame_count = 0
    with open(data_path, 'wb') as f:
        writer = Writer(f, fields)

        while (data := data_queue.get()) is not None:
            frame, controller_states, info = data
            frame = cv2.resize(frame, (width, height))
            av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            packet = stream.encode(av_frame)
            container.mux(packet)

            joystick = [float(getattr(controller_states[0], field)) / 127 for field in M64pButtons.get_joystick_fields()]
            buttons = [int(getattr(controller_states[0], field)) for field in M64pButtons.get_button_fields()]
            info = {f'info.{name}': np.array(info[name], dtype=dtype).reshape(shape) for name, dtype, shape in info_fields}
            writer.writerow({
                'frame_index': np.array(frame_count, dtype=np.uint32),
                'action.joystick': np.array(joystick, dtype=np.float32),
                'action.buttons': np.array(buttons, dtype=np.uint8),
                **info,
            })

            frame_count += 1

    packet = stream.encode(None)
    container.mux(packet)
    container.close()
