"""
Shared, atomic-write-with-retry checkpoint I/O used by every learnable
component's own store_weights()/load_weights() method (encoders, experts,
registries -- grep each class for "atomic_torch_save" to see where).

A plain torch.save is not atomic and not resilient to transient filesystem
hiccups: a write failure partway through can leave a half-written file
sitting at the real target path, and has been observed in practice
(PytorchStreamWriter "file write failed" on a flaky GPFS mount) to leave the
whole process stuck rather than cleanly raising, hanging a job for its
entire wall-clock allocation instead of failing fast. Writing to a temp file
and renaming into place means a failed attempt never produces a half-written
file at the real path, and retrying with a short backoff rides out a
transient glitch instead of taking the whole run down with it.

enc_pre.py / text_to_text.py / ocr_to_text.py duplicate an equivalent
``_atomic_torch_save`` for their own resumable "wrapper" checkpoints (full
training state: optimizer, RNG, epoch/batch counters -- not a single
component's weights, so it doesn't fit a component-level store_weights
call); this module is the encoder/expert/registry-level counterpart, used
by each component's *own* store_weights so callers can save/load a single
component (e.g. just the text-output expert) through its own canonical
method instead of duplicating the equivalent logic ad hoc.
"""
import os
import time

import torch


def atomic_torch_save(obj, path, retries=3, backoff_seconds=5.0):
    tmp_path = f"{path}.tmp{os.getpid()}"
    last_err = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries - 1:
                print(f"[checkpoint] write to {path} failed ({e}); "
                      f"retrying ({attempt + 1}/{retries})...")
                time.sleep(backoff_seconds)
    raise last_err
