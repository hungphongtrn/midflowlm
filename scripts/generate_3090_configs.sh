#!/bin/bash
# Generate 3090 configs from v0_1_matrix configs
# Run this on the server after pulling the latest code

set -e

SOURCE_DIR="configs/v0_1_matrix"
TARGET_DIR="configs/v0_1_matrix_3090"

echo "Creating 3090 configs in $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

for config in "$SOURCE_DIR"/*.yaml; do
    filename=$(basename "$config")
    target="$TARGET_DIR/$filename"
    
    echo "Processing: $filename"
    
    # Read and modify the config
    python3 << EOF
import yaml
import sys

with open('$config') as f:
    config = yaml.safe_load(f)

# Update experiment name
orig_name = config['experiment_name']
config['experiment_name'] = orig_name + '_3090'

# Update cache dir
if 'cache_dir' in config.get('teacher_cache', {}):
    config['teacher_cache']['cache_dir'] = config['teacher_cache']['cache_dir'].replace(orig_name, config['experiment_name'])

# Reduce batch size for 3090
config['data']['batch_size'] = 2
config['data']['num_workers'] = 2

# Update grad accumulation to maintain effective batch size
# Original: bs=3, accum=5 -> effective=15
# New: bs=2, accum=8 -> effective=16 (close enough)
config['train_loop']['accumulate_grad_batches'] = 8

# Update checkpoint and log dirs
orig_ckpt = config['train_loop']['checkpoint_dir']
config['train_loop']['checkpoint_dir'] = orig_ckpt.replace(orig_name, config['experiment_name'])

orig_log = config['logging']['log_dir']
config['logging']['log_dir'] = orig_log.replace(orig_name, config['experiment_name'])

if config['logging'].get('tensorboard', {}).get('log_dir'):
    orig_tb = config['logging']['tensorboard']['log_dir']
    config['logging']['tensorboard']['log_dir'] = orig_tb.replace(orig_name, config['experiment_name'])

# Update wandb project
config['wandb']['project'] = 'midflowlm-v0-1-3090'

# Add 3090 tag
if 'tags' in config['wandb']:
    if '3090' not in config['wandb']['tags']:
        config['wandb']['tags'].append('3090')

# Write with header
with open('$target', 'w') as f:
    f.write(f"# {config['experiment_name']}\\n")
    f.write(f"# 3090-adapted version of {orig_name}.yaml\\n")
    f.write(f"# Changes: batch_size=2, grad_accum=8, num_workers=2\\n")
    f.write(f"#\\n\\n")
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created: $target")
EOF
done

echo ""
echo "All 3090 configs created in $TARGET_DIR"
echo "Key changes: batch_size=2, num_workers=2, grad_accum=8"
