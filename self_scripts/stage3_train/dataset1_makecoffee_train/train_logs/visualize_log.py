#!/usr/bin/env python3
"""
Training Log Visualization Script
Parse and visualize training metrics from log files
"""

import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path

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

def parse_log_line(line):
    """Parse a single log line and extract all metrics"""
    if 'step:' not in line:
        return None

    result = {}

    # Extract values using regex
    patterns = {
        'step': r'step:(\S+)',
        'sample': r'smpl:(\S+)',
        'episode': r'ep:(\S+)',
        'epoch': r'epch:(\S+)',
        'loss': r'loss:(\S+)',
        'gradient': r'grdn:(\S+)',
        'lr': r'lr:(\S+)',
        'update_time': r'updt_s:(\S+)',
        'data_time': r'data_s:(\S+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            result[key] = parse_value(match.group(1))
        else:
            result[key] = np.nan

    return result

def load_log_data(log_text):
    """Load and parse log data from text"""
    data = {
        'steps': [],
        'samples': [],
        'episodes': [],
        'epochs': [],
        'losses': [],
        'gradients': [],
        'learning_rates': [],
        'update_times': [],
        'data_times': []
    }

    for line in log_text.strip().split('\n'):
        parsed = parse_log_line(line)
        if parsed:
            data['steps'].append(parsed['step'])
            data['samples'].append(parsed['sample'])
            data['episodes'].append(parsed['episode'])
            data['epochs'].append(parsed['epoch'])
            data['losses'].append(parsed['loss'])
            data['gradients'].append(parsed['gradient'])
            data['learning_rates'].append(parsed['lr'])
            data['update_times'].append(parsed['update_time'])
            data['data_times'].append(parsed['data_time'])

    return data

def create_visualization(data, output_path='training_visualization.png'):
    """Create comprehensive visualization of training metrics"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Training Metrics Visualization', fontsize=18, fontweight='bold', y=0.995)

    # Plot 1: Loss over steps (larger plot)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data['steps'], data['losses'], 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # Plot 2: Gradient norm (with outlier handling)
    ax2 = fig.add_subplot(gs[0, 2])
    # Filter out nan values for better visualization
    valid_grads = [(s, g) for s, g in zip(data['steps'], data['gradients']) if not np.isnan(g)]
    if valid_grads:
        steps_g, grads_g = zip(*valid_grads)
        ax2.plot(steps_g, grads_g, 'r-', linewidth=1.2, alpha=0.7)
    ax2.set_xlabel('Steps', fontsize=10)
    ax2.set_ylabel('Gradient Norm', fontsize=10)
    ax2.set_title('Gradient Norm', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Epochs progress
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(data['steps'], data['epochs'], 'g-', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Steps', fontsize=10)
    ax3.set_ylabel('Epochs', fontsize=10)
    ax3.set_title('Training Progress', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Timing metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(data['steps'], data['update_times'], 'purple', linewidth=1.2,
             alpha=0.7, label='Update Time')
    ax4.plot(data['steps'], data['data_times'], 'orange', linewidth=1.2,
             alpha=0.7, label='Data Time')
    ax4.set_xlabel('Steps', fontsize=10)
    ax4.set_ylabel('Time (s)', fontsize=10)
    ax4.set_title('Update & Data Loading Time', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Plot 5: Samples processed
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(data['steps'], np.array(data['samples'])/1e6, 'cyan', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Steps', fontsize=10)
    ax5.set_ylabel('Samples (M)', fontsize=10)
    ax5.set_title('Total Samples', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Plot 6: Episodes
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(data['steps'], np.array(data['episodes'])/1000, 'magenta', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Steps', fontsize=10)
    ax6.set_ylabel('Episodes (K)', fontsize=10)
    ax6.set_title('Episodes Processed', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Plot 7: Loss trend with moving average
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.plot(data['steps'], data['losses'], 'b-', linewidth=1, alpha=0.3, label='Loss')

    # Calculate moving average
    window = min(20, len(data['losses']) // 10)
    if window > 1:
        ma_losses = np.convolve(data['losses'], np.ones(window)/window, mode='valid')
        ma_steps = data['steps'][(window-1)//2:-(window//2)] if window % 2 == 0 else data['steps'][window//2:-window//2+1]
        ax7.plot(ma_steps, ma_losses, 'darkblue', linewidth=2, label=f'MA({window})')

    ax7.set_xlabel('Steps', fontsize=10)
    ax7.set_ylabel('Loss', fontsize=10)
    ax7.set_title('Loss with Moving Average', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_xlim(left=0)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    print(f"Total Steps: {int(data['steps'][-1]):,}")
    print(f"Total Samples: {int(data['samples'][-1]):,}")
    print(f"Total Episodes: {int(data['episodes'][-1]):,}")
    print(f"Final Epoch: {data['epochs'][-1]:.2f}")
    print(f"Initial Loss: {data['losses'][0]:.4f}")
    print(f"Final Loss: {data['losses'][-1]:.4f}")
    print(f"Loss Reduction: {((data['losses'][0] - data['losses'][-1]) / data['losses'][0] * 100):.1f}%")
    print(f"Avg Update Time: {np.nanmean(data['update_times']):.3f}s")
    print(f"Avg Data Time: {np.nanmean(data['data_times']):.3f}s")
    print("="*50)

    return fig

# Main execution
if __name__ == "__main__":
    # You can either paste log data here or read from a file
    log_file_path = "training.log"  # Change this to your log file path

    # Try to read from file first, otherwise use embedded data
    if Path(log_file_path).exists():
        print(f"Reading log from: {log_file_path}")
        with open(log_file_path, 'r') as f:
            log_text = f.read()
    else:
        # Embedded log data - replace with your actual log
        print("Using embedded log data...")
        log_text = """INFO 2026-08-07 18:27:47 ot_train.py:423 step:200 smpl:102K ep:115 epch:2.86 loss:0.985 grdn:nan lr:5.0e-05 updt_s:0.331 data_s:0.028
INFO 2026-08-07 18:28:57 ot_train.py:423 step:400 smpl:205K ep:229 epch:5.73 loss:0.367 grdn:8.312 lr:5.0e-05 updt_s:0.334 data_s:0.019"""

    # Parse data
    print("Parsing log data...")
    data = load_log_data(log_text)
    print(f"Parsed {len(data['steps'])} data points")

    # Create visualization
    print("Creating visualization...")
    fig = create_visualization(data, output_path='/home/ksa/lerobot/training_visualization.png')

    print("\n✓ Done!")
