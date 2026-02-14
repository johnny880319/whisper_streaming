# Ray Whisper Streaming
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Scalable, concurrent real-time speech recognition based on [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) and [Ray](https://github.com/ray-project/ray).

## Why This Fork?
Original whisper_streaming is excellent for single-stream processing. However, when building a voice bot or meeting assistant serving multiple concurrent users, simply spawning threads isn't enough.

This project introduces a Ray-based Actor-Worker architecture to:

- Load Balance: Distribute ASR jobs from multiple streams to a shared pool of GPU workers.

- Isolate Resources: Decouple the lightweight streaming logic (CPU) from the heavy model inference (GPU).

- Scale Horizontally: Easily scale from 1 to 100 streams by adding more GPU workers.


## Architecture
1. OnlineASRProcessor: Handle buffering, and hypothesis alignment logic (Stateful, CPU-bound).

2. Global Queue: Acts as a load balancer.

3. FasterWhisperWorker: Consumes audio features and runs inference (Stateless, GPU-bound).

## Environment Setup

```bash
# Recommended: Use uv for fast dependency management
uv sync

# If you want to use the exact same dependency versions as the author, you can also use
uv sync --lock

# Or standard pip
pip install -e .
```

## Running the Script

Since uv run is not compatible with Ray, you need to run your script with python -m to make sure the dependencies are correctly loaded in the Ray workers.

```bash
source .venv/bin/activate
python -m path.to.your.script
```

## Usage

1. Create one `FasterWhisperWorkerCluster`.
2. Get one shared `global_asr_queue`.
3. Build multiple `OnlineASRProcessor(RayFasterWhisperASR(...))` instances, one per stream. You can create them locally or remotely, depending on your resource constraints like CPU.
4. Stream audio chunks using the `OnlineASRProcessor` API (`insert_audio_chunk`, `process_iter`, `finish`), and the ASR jobs will be automatically load balanced to the worker cluster.

```python
# set CUDA_VISIBLE_DEVICES to specify which GPUs to use.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# import used modules
from pathlib import Path

import librosa
import numpy as np

from whisper_online import FasterWhisperWorkerCluster, OnlineASRProcessor, RayFasterWhisperASR

# create and run the Faster Whisper Worker Cluster.
# The worker count will be automatically set to the number of GPUs specified in CUDA_VISIBLE_DEVICES, but you can also set a max_worker_count to limit it.
faster_whisper_cluster = FasterWhisperWorkerCluster()
faster_whisper_cluster.run()

# The ASR message queue that load balances the ASR jobs to the whisper workers
global_asr_queue = faster_whisper_cluster.get_global_asr_queue()

# prepare your audio stream, here we just load some local audio files as examples.
audio_files = Path("path/to/your/audio/files").glob("*.wav")
audio_stream = [
    librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)[0]
    for i, audio_path in enumerate(audio_files)
]

# For simplification, we create OnlineASRProcessor locally.
# However, you can also create OnlineASRProcessor(RayFasterWhisperASR(...)) remotely by passing the global_asr_queue.
# Which can save CPU resources.
online_processors = [
    OnlineASRProcessor(RayFasterWhisperASR(global_asr_queue=global_asr_queue))
    for _ in range(len(audio_stream))
]

# Now you can use the online_processors to process the audio stream in a streaming way.
CHUNK_SECOND = 0.25
all_finished = False
streaming_time = 0.0
while not all_finished:
    all_finished = True
    start_sample, end_sample = int(streaming_time * 16000), int((streaming_time + CHUNK_SECOND) * 16000)
    for i, processor in enumerate(online_processors):
        # skip if the audio stream is already finished
        if start_sample >= len(audio_stream[i]):
            continue
        all_finished = False

        # process the current audio chunk
        processor.insert_audio_chunk(audio_stream[i][start_sample:end_sample])
        start_timestamp, end_timestamp, transcript = processor.process_iter()
        # do_something(start_timestamp, end_timestamp, transcript)

        # finish the ASR job if the audio stream is finished.
        if end_sample >= len(audio_stream[i]):
            start_timestamp, end_timestamp, transcript = processor.finish()
            # do_something(start_timestamp, end_timestamp, transcript)

    streaming_time += CHUNK_SECOND
```

## Configuration
You can configure the FasterWhisperWorkerCluster with following parameters:

| Parameter | Description | Default |
| --- | --- | --- |
| `lan` | The language for the ASR model. | `"zh"` |
| `model_size_or_path` | The model size or path to load the Faster Whisper model. | `"large-v2"` |
| `max_worker_count` | The maximum number of workers to use. The actual worker count will be the minimum of this value and the number of available GPUs. If not specified, it will be set to the number of GPUs available. | `None` |

---
