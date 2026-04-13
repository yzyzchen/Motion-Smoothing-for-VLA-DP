import numpy as np
import matplotlib.pyplot as plt


def ema_filter(x, epsilon):
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = (1 - epsilon) * y[t - 1] + epsilon * x[t]
    return y


def minimum_jerk(start, end, length):
    tau = np.linspace(0.0, 1.0, length)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return start + (end - start) * s


def smooth_noise(length, scale=0.002, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, scale, size=length)
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(x, (2, 2), mode='edge')
    y = np.convolve(padded, kernel, mode='valid')
    return y


def make_realistic_chunk(start, end, length, noise_scale=0.0015, bump_scale=0.0035, seed=None):
    base = minimum_jerk(start, end, length)
    tau = np.linspace(0.0, 1.0, length)

    rng = np.random.default_rng(seed)
    bump_center = rng.uniform(0.45, 0.75)
    bump_width = rng.uniform(0.06, 0.12)
    bump_sign = rng.choice([-1.0, 1.0])

    bump = bump_sign * bump_scale * np.exp(-((tau - bump_center) ** 2) / (2 * bump_width**2))
    noise = smooth_noise(length, scale=noise_scale, seed=seed)

    return base + bump + noise


def linear_blend(old_tail, new_head, blend_len):
    """
    Blend old_tail[k] with new_head[k] over k = 0...blend_len-1:
        blended[k] = (1 - alpha_k) * old_tail[k] + alpha_k * new_head[k]
    """
    blended = np.zeros(blend_len)
    for k in range(blend_len):
        alpha = 1.0 if blend_len == 1 else k / (blend_len - 1)
        blended[k] = (1 - alpha) * old_tail[k] + alpha * new_head[k]
    return blended


def build_stitched_trajectory(chunks, dt, blend_len, delay_steps):
    segments = []
    zones = []
    chunk_times = []
    current_time = 0.0

    def append_segment(values, name=None):
        nonlocal current_time
        values = np.asarray(values)
        start = current_time
        end = current_time + len(values) * dt
        segments.append(values)
        current_time = end
        if name is not None:
            zones.append((name, start, end))
        return start, end

    first_chunk = chunks[0]
    first_main = first_chunk[:-blend_len]
    append_segment(first_main)
    chunk_times.append(np.arange(len(first_chunk)) * dt)
    previous_output = first_main[-1]

    for idx in range(1, len(chunks)):
        delay_vals = np.full(delay_steps, previous_output)
        append_segment(delay_vals, name=f"delay{idx}")

        delay_end = zones[-1][2]
        chunk_times.append(delay_end + np.arange(len(chunks[idx])) * dt)

        held_old = np.full(blend_len, delay_vals[-1])
        blended_vals = linear_blend(held_old, chunks[idx][:blend_len], blend_len)
        append_segment(blended_vals, name=f"blend{idx}{idx + 1}")

        if idx < len(chunks) - 1:
            middle = chunks[idx][blend_len:-blend_len]
        else:
            middle = chunks[idx][blend_len:]

        append_segment(middle)
        previous_output = middle[-1] if len(middle) > 0 else blended_vals[-1]

    stitched = np.concatenate(segments)
    t = np.arange(len(stitched)) * dt
    return stitched, t, zones, chunk_times


# =========================
# Parameters
# =========================
dt = 0.02
chunk_len = 50
blend_len = 10
delay_steps = 5
epsilons = [0.2, 0.4]

# =========================
# Raw chunks
# =========================
chunk1 = make_realistic_chunk(-0.16, -0.11, chunk_len, seed=1)
chunk2 = make_realistic_chunk(-0.08, -0.02, chunk_len, seed=2) + 0.005
chunk3 = make_realistic_chunk(-0.10, -0.15, chunk_len, seed=3)
chunk4 = make_realistic_chunk(-0.14, -0.06, chunk_len, seed=4) - 0.003

chunks = [chunk1, chunk2, chunk3, chunk4]
stitched, t, zones, chunk_times = build_stitched_trajectory(chunks, dt, blend_len, delay_steps)

# =========================
# Plot
# =========================
fig, axes = plt.subplots(len(epsilons), 1, figsize=(14, 12), sharex=True)

for ax, epsilon in zip(axes, epsilons):
    optimized = ema_filter(stitched, epsilon)

    for idx, (chunk_t, chunk_vals) in enumerate(zip(chunk_times, chunks), start=1):
        ax.plot(chunk_t, chunk_vals, ':', linewidth=2.0, label=f'Raw Action Chunk {idx}')

    ax.plot(t, stitched, '--', linewidth=2.8, label='Linearly Blended Action')
    ax.plot(t, optimized, linewidth=7, alpha=0.12, solid_capstyle='round')
    ax.plot(t, optimized, linewidth=3.2, solid_capstyle='round', label='Optimized Action')

    first_blend = True
    first_delay = True
    for name, start, end in zones:
        if "blend" in name:
            ax.axvspan(
                start, end,
                alpha=0.18,
                label='Blending Zone' if first_blend else None
            )
            first_blend = False
        elif "delay" in name:
            ax.axvspan(
                start, end,
                alpha=0.10,
                label='Inference Delay' if first_delay else None
            )
            first_delay = False

    ax.set_ylabel('Joint Angle (rad)')
    ax.set_title(f'Motion Smoothing (epsilon={epsilon})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
