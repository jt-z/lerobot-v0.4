import matplotlib.pyplot as plt
import re
import numpy as np

# Read log file
log_file = '/home/ksa/lerobot/self_scripts/stage3_train/train_logs/smolVLA.log'

with open(log_file, 'r') as f:
    log_data = f.read()

# Parse the log data
pattern = r'INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?step:(\d+)K.*?loss:([\d.]+).*?grdn:([\d.]+|inf).*?lr:([\d.e-]+).*?updt_s:([\d.]+)'

matches = re.findall(pattern, log_data)

# Extract data
timestamps = []
steps = []
losses = []
gradients = []
learning_rates = []
update_times = []

for match in matches:
    timestamp, step, loss, grdn, lr, updt_s = match
    timestamps.append(timestamp)
    steps.append(int(step))
    losses.append(float(loss))
    # Handle 'inf' gradient values
    if grdn == 'inf':
        gradients.append(np.nan)  # Use NaN for inf values
    else:
        gradients.append(float(grdn))
    learning_rates.append(float(lr))
    update_times.append(float(updt_s))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('smolVLA Training Metrics', fontsize=16, fontweight='bold')

# Plot 1: Loss over steps
axes[0, 0].plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7)
axes[0, 0].set_xlabel('Training Steps (K)', fontsize=11)
axes[0, 0].set_ylabel('Loss', fontsize=11)
axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim([min(steps), max(steps)])

# Plot 2: Gradient norm over steps (excluding inf values)
valid_gradients = [g for g in gradients if not np.isnan(g)]
valid_steps_for_grad = [s for s, g in zip(steps, gradients) if not np.isnan(g)]
axes[0, 1].plot(valid_steps_for_grad, valid_gradients, 'g-', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Training Steps (K)', fontsize=11)
axes[0, 1].set_ylabel('Gradient Norm', fontsize=11)
axes[0, 1].set_title('Gradient Norm (inf values excluded)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([min(steps), max(steps)])

# Plot 3: Learning rate over steps
axes[1, 0].plot(steps, learning_rates, 'r-', linewidth=1.5, alpha=0.7)
axes[1, 0].set_xlabel('Training Steps (K)', fontsize=11)
axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([min(steps), max(steps)])
axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Plot 4: Update time over steps
axes[1, 1].plot(steps, update_times, 'orange', linewidth=1.5, alpha=0.7)
axes[1, 1].set_xlabel('Training Steps (K)', fontsize=11)
axes[1, 1].set_ylabel('Update Time (s)', fontsize=11)
axes[1, 1].set_title('Update Time per Step', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([min(steps), max(steps)])

# Add statistics text box
stats_text = f"""Training Statistics:
Steps: {min(steps)}K → {max(steps)}K
Loss: {min(losses):.4f} → {losses[-1]:.4f}
Avg Gradient: {np.nanmean(gradients):.3f}
Avg Update Time: {np.mean(update_times):.3f}s
Learning Rate: {learning_rates[0]:.2e}"""

fig.text(0.02, 0.02, stats_text, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='bottom', family='monospace')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# Save the figure
output_file = '/home/ksa/lerobot/self_scripts/stage3_train/train_logs/smolVLA_training_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'Visualization saved to: {output_file}')

# Also display summary
print(f'\n=== Training Summary ===')
print(f'Total steps: {min(steps)}K - {max(steps)}K ({max(steps) - min(steps)}K steps)')
print(f'Loss trend: {losses[0]:.4f} → {losses[-1]:.4f} (change: {losses[-1] - losses[0]:+.4f})')
print(f'Average gradient norm: {np.nanmean(gradients):.3f}')
print(f'Gradient norm range: {np.nanmin(gradients):.3f} - {np.nanmax(gradients):.3f}')
print(f'Learning rate: {learning_rates[0]:.2e} (constant)')
print(f'Average update time: {np.mean(update_times):.3f}s')
print(f'Update time range: {min(update_times):.3f}s - {max(update_times):.3f}s')
print(f'Total data points: {len(steps)}')

plt.show()
