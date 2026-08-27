import matplotlib.pyplot as plt
import re
import numpy as np

# Training log data - paste your complete log here
log_data = """INFO 2026-08-07 18:27:47 ot_train.py:423 step:200 smpl:102K ep:115 epch:2.86 loss:0.985 grdn:nan lr:5.0e-05 updt_s:0.331 data_s:0.028
INFO 2026-08-07 18:28:57 ot_train.py:423 step:400 smpl:205K ep:229 epch:5.73 loss:0.367 grdn:8.312 lr:5.0e-05 updt_s:0.334 data_s:0.019
INFO 2026-08-07 18:30:08 ot_train.py:423 step:600 smpl:307K ep:344 epch:8.59 loss:0.281 grdn:6.877 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-07 18:31:18 ot_train.py:423 step:800 smpl:410K ep:458 epch:11.45 loss:0.239 grdn:6.173 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-07 18:32:27 ot_train.py:423 step:1K smpl:512K ep:573 epch:14.31 loss:0.214 grdn:5.656 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-07 18:33:38 ot_train.py:423 step:1K smpl:614K ep:687 epch:17.18 loss:0.196 grdn:5.361 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-07 18:34:47 ot_train.py:423 step:1K smpl:717K ep:802 epch:20.04 loss:0.178 grdn:5.176 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 18:35:57 ot_train.py:423 step:2K smpl:819K ep:916 epch:22.90 loss:0.170 grdn:4.703 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-07 18:37:07 ot_train.py:423 step:2K smpl:922K ep:1K epch:25.77 loss:0.159 grdn:4.472 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-07 18:38:17 ot_train.py:423 step:2K smpl:1M ep:1K epch:28.63 loss:0.150 grdn:4.218 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-07 18:39:27 ot_train.py:423 step:2K smpl:1M ep:1K epch:31.49 loss:0.142 grdn:3.804 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 18:40:38 ot_train.py:423 step:2K smpl:1M ep:1K epch:34.35 loss:0.139 grdn:3.966 lr:5.0e-05 updt_s:0.329 data_s:0.027
INFO 2026-08-07 18:41:48 ot_train.py:423 step:3K smpl:1M ep:1K epch:37.22 loss:0.136 grdn:3.679 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-07 18:42:58 ot_train.py:423 step:3K smpl:1M ep:2K epch:40.08 loss:0.128 grdn:3.479 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-07 18:44:07 ot_train.py:423 step:3K smpl:2M ep:2K epch:42.94 loss:0.125 grdn:3.249 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 18:45:16 ot_train.py:423 step:3K smpl:2M ep:2K epch:45.81 loss:0.122 grdn:3.419 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 18:46:27 ot_train.py:423 step:3K smpl:2M ep:2K epch:48.67 loss:0.115 grdn:3.072 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-07 18:47:37 ot_train.py:423 step:4K smpl:2M ep:2K epch:51.53 loss:0.115 grdn:2.918 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-07 18:48:46 ot_train.py:423 step:4K smpl:2M ep:2K epch:54.39 loss:0.111 grdn:2.896 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 18:49:57 ot_train.py:423 step:4K smpl:2M ep:2K epch:57.26 loss:0.105 grdn:2.695 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-07 18:51:07 ot_train.py:423 step:4K smpl:2M ep:2K epch:60.12 loss:0.106 grdn:2.764 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-07 18:52:16 ot_train.py:423 step:4K smpl:2M ep:3K epch:62.98 loss:0.105 grdn:2.772 lr:5.0e-05 updt_s:0.326 data_s:0.020
INFO 2026-08-07 18:53:27 ot_train.py:423 step:5K smpl:2M ep:3K epch:65.84 loss:0.100 grdn:2.736 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-07 18:54:37 ot_train.py:423 step:5K smpl:2M ep:3K epch:68.71 loss:0.097 grdn:2.396 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-07 18:55:46 ot_train.py:423 step:5K smpl:3M ep:3K epch:71.57 loss:0.096 grdn:2.326 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 18:56:57 ot_train.py:423 step:5K smpl:3M ep:3K epch:74.43 loss:0.094 grdn:2.448 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-07 18:58:06 ot_train.py:423 step:5K smpl:3M ep:3K epch:77.30 loss:0.093 grdn:2.336 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 18:59:16 ot_train.py:423 step:6K smpl:3M ep:3K epch:80.16 loss:0.091 grdn:2.287 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-07 19:00:26 ot_train.py:423 step:6K smpl:3M ep:3K epch:83.02 loss:0.088 grdn:2.196 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-07 19:01:35 ot_train.py:423 step:6K smpl:3M ep:3K epch:85.88 loss:0.087 grdn:2.264 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 19:02:47 ot_train.py:423 step:6K smpl:3M ep:4K epch:88.75 loss:0.086 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-07 19:03:56 ot_train.py:423 step:6K smpl:3M ep:4K epch:91.61 loss:0.085 grdn:2.140 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 19:05:05 ot_train.py:423 step:7K smpl:3M ep:4K epch:94.47 loss:0.082 grdn:2.049 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-07 19:06:15 ot_train.py:423 step:7K smpl:3M ep:4K epch:97.34 loss:0.082 grdn:2.020 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-07 19:07:25 ot_train.py:423 step:7K smpl:4M ep:4K epch:100.20 loss:0.082 grdn:1.951 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-07 19:08:34 ot_train.py:423 step:7K smpl:4M ep:4K epch:103.06 loss:0.081 grdn:2.096 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 19:09:45 ot_train.py:423 step:7K smpl:4M ep:4K epch:105.92 loss:0.079 grdn:2.077 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-07 19:10:54 ot_train.py:423 step:8K smpl:4M ep:4K epch:108.79 loss:0.077 grdn:1.818 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-07 19:12:04 ot_train.py:423 step:8K smpl:4M ep:4K epch:111.65 loss:0.075 grdn:1.757 lr:5.0e-05 updt_s:0.326 data_s:0.020
INFO 2026-08-07 19:13:15 ot_train.py:423 step:8K smpl:4M ep:5K epch:114.51 loss:0.076 grdn:1.997 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-07 19:14:25 ot_train.py:423 step:8K smpl:4M ep:5K epch:117.38 loss:0.073 grdn:inf lr:5.0e-05 updt_s:0.331 data_s:0.019
INFO 2026-08-07 19:15:35 ot_train.py:423 step:8K smpl:4M ep:5K epch:120.24 loss:0.074 grdn:1.866 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-07 19:16:45 ot_train.py:423 step:9K smpl:4M ep:5K epch:123.10 loss:0.073 grdn:1.728 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-07 19:17:54 ot_train.py:423 step:9K smpl:5M ep:5K epch:125.96 loss:0.071 grdn:1.622 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-07 19:19:05 ot_train.py:423 step:9K smpl:5M ep:5K epch:128.83 loss:0.069 grdn:1.704 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-07 19:20:14 ot_train.py:423 step:9K smpl:5M ep:5K epch:131.69 loss:0.070 grdn:1.528 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 19:21:23 ot_train.py:423 step:9K smpl:5M ep:5K epch:134.55 loss:0.068 grdn:1.633 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-07 19:22:34 ot_train.py:423 step:10K smpl:5M ep:5K epch:137.42 loss:0.067 grdn:1.606 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-07 19:23:44 ot_train.py:423 step:10K smpl:5M ep:6K epch:140.28 loss:0.067 grdn:1.558 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-07 19:24:53 ot_train.py:423 step:10K smpl:5M ep:6K epch:143.14 loss:0.065 grdn:1.518 lr:5.0e-05 updt_s:0.325 data_s:0.019"""

# Parse the log data
def parse_value(val_str):
    """Parse values like '1K', '1M', 'nan', 'inf' to numeric values"""
    val_str = val_str.strip()
    if val_str == 'nan':
        return np.nan
    if val_str == 'inf':
        return np.nan  # Treat inf as nan for plotting

    if 'K' in val_str:
        return float(val_str.replace('K', '')) * 1000
    elif 'M' in val_str:
        return float(val_str.replace('M', '')) * 1000000
    else:
        try:
            return float(val_str)
        except:
            return np.nan

# Initialize lists to store parsed data
steps = []
samples = []
episodes = []
epochs = []
losses = []
gradients = []
learning_rates = []
update_times = []
data_times = []

# Parse each line
for line in log_data.strip().split('\n'):
    if 'step:' not in line:
        continue

    # Extract values using regex
    step_match = re.search(r'step:(\S+)', line)
    smpl_match = re.search(r'smpl:(\S+)', line)
    ep_match = re.search(r'ep:(\S+)', line)
    epch_match = re.search(r'epch:(\S+)', line)
    loss_match = re.search(r'loss:(\S+)', line)
    grdn_match = re.search(r'grdn:(\S+)', line)
    lr_match = re.search(r'lr:(\S+)', line)
    updt_match = re.search(r'updt_s:(\S+)', line)
    data_match = re.search(r'data_s:(\S+)', line)

    if step_match:
        steps.append(parse_value(step_match.group(1)))
        samples.append(parse_value(smpl_match.group(1)) if smpl_match else np.nan)
        episodes.append(parse_value(ep_match.group(1)) if ep_match else np.nan)
        epochs.append(parse_value(epch_match.group(1)) if epch_match else np.nan)
        losses.append(parse_value(loss_match.group(1)) if loss_match else np.nan)
        gradients.append(parse_value(grdn_match.group(1)) if grdn_match else np.nan)
        learning_rates.append(parse_value(lr_match.group(1)) if lr_match else np.nan)
        update_times.append(parse_value(updt_match.group(1)) if updt_match else np.nan)
        data_times.append(parse_value(data_match.group(1)) if data_match else np.nan)

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Training Metrics Visualization', fontsize=16, fontweight='bold')

# Plot 1: Loss over steps
axes[0, 0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=3)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Gradient norm over steps
axes[0, 1].plot(steps, gradients, 'r-', linewidth=2, marker='o', markersize=3)
axes[0, 1].set_xlabel('Training Steps')
axes[0, 1].set_ylabel('Gradient Norm')
axes[0, 1].set_title('Gradient Norm')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Epochs over steps
axes[1, 0].plot(steps, epochs, 'g-', linewidth=2, marker='o', markersize=3)
axes[1, 0].set_xlabel('Training Steps')
axes[1, 0].set_ylabel('Epochs')
axes[1, 0].set_title('Training Progress (Epochs)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Update time and data time
axes[1, 1].plot(steps, update_times, 'purple', linewidth=2, marker='o', markersize=3, label='Update Time')
axes[1, 1].plot(steps, data_times, 'orange', linewidth=2, marker='s', markersize=3, label='Data Time')
axes[1, 1].set_xlabel('Training Steps')
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('Update and Data Loading Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Samples processed over steps
axes[2, 0].plot(steps, samples, 'cyan', linewidth=2, marker='o', markersize=3)
axes[2, 0].set_xlabel('Training Steps')
axes[2, 0].set_ylabel('Samples Processed')
axes[2, 0].set_title('Total Samples Processed')
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Episodes over steps
axes[2, 1].plot(steps, episodes, 'magenta', linewidth=2, marker='o', markersize=3)
axes[2, 1].set_xlabel('Training Steps')
axes[2, 1].set_ylabel('Episodes')
axes[2, 1].set_title('Episodes Processed')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ksa/lerobot/training_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: /home/ksa/lerobot/training_visualization.png")
plt.show()
