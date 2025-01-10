import numpy as np
import tensorflow as tf
import secrets

def spec_augment(
    segments,
    num_time_masks=2,
    time_mask_size=20,
    num_freq_masks=2,
    freq_mask_size=10
):
    """
    Apply SpecAugment-style time and frequency masking to a batch of spectrogram segments.
    segments shape: (S, freq_dim, time_dim, channels)
    """
    # Convert to NumPy if it's a TF tensor
    if isinstance(segments, tf.Tensor):
        segments = segments.numpy()
    
    S, freq_dim, time_dim, _ = segments.shape

    for i in range(S):
        segment = segments[i]  # shape: (freq_dim, time_dim, 1) in NumPy

        # ---- Frequency Masking ----
        for _ in range(num_freq_masks):
            mask_size = secrets.SystemRandom().randint(0, freq_mask_size)
            f0 = secrets.SystemRandom().randint(0, max(0, freq_dim - mask_size))
            segment[f0 : f0 + mask_size, :, :] = 0  # zero out in-place

        # ---- Time Masking ----
        for _ in range(num_time_masks):
            mask_size = secrets.SystemRandom().randint(0, time_mask_size)
            t0 = secrets.SystemRandom().randint(0, max(0, time_dim - mask_size))
            segment[:, t0 : t0 + mask_size, :] = 0  # zero out in-place

    # Convert back to tf.Tensor if needed
    segments = tf.convert_to_tensor(segments)
    return segments
