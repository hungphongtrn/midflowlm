#!/bin/bash
# Quick fix script to split existing cache into train/val folders
# Val set = 5% (1000 samples), Train set = 95% (19000 samples)

set -e

CACHE_DIR="./cache/tinystories_qwen_boundary_states"
VAL_SIZE=1000
TRAIN_SIZE=19000

echo "=== Splitting cache into train/val folders ==="
echo "Cache dir: $CACHE_DIR"
echo "Val size: $VAL_SIZE samples"
echo "Train size: $TRAIN_SIZE samples"

# Create train and val directories
mkdir -p "$CACHE_DIR/train"
mkdir -p "$CACHE_DIR/val"

echo ""
echo "=== Moving first $VAL_SIZE shards to val/ ==="
for i in $(seq -w 0 999); do
    shard="shard_${i}_of_20000.safetensors"
    if [ -f "$CACHE_DIR/$shard" ]; then
        mv "$CACHE_DIR/$shard" "$CACHE_DIR/val/"
    fi
done

echo "=== Moving remaining $TRAIN_SIZE shards to train/ ==="
for i in $(seq -w 1000 19999); do
    shard="shard_${i}_of_20000.safetensors"
    if [ -f "$CACHE_DIR/$shard" ]; then
        mv "$CACHE_DIR/$shard" "$CACHE_DIR/train/"
    fi
done

echo ""
echo "=== Renaming val shards (0-999 of 1000) ==="
cd "$CACHE_DIR/val"
for i in $(seq -w 0 999); do
    old_name="shard_${i}_of_20000.safetensors"
    new_name="shard_${i}_of_1000.safetensors"
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
    fi
done
cd - > /dev/null

echo "=== Renaming train shards (0-18999 of 19000) ==="
cd "$CACHE_DIR/train"
for i in $(seq -w 1000 19999); do
    # Calculate new index (0-18999)
    new_idx=$(printf "%04d" $((10#$i - 1000)))
    old_name="shard_${i}_of_20000.safetensors"
    new_name="shard_${new_idx}_of_19000.safetensors"
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
    fi
done
cd - > /dev/null

echo ""
echo "=== Creating metadata.json for val ==="
cat > "$CACHE_DIR/val/metadata.json" << 'EOF'
{
  "model_name": "Qwen/Qwen3.5-0.8B",
  "model_revision": null,
  "start_layer": 8,
  "end_layer": 11,
  "span_depth": 4,
  "seq_len": 128,
  "store_logits": false,
  "num_samples": 1000
}
EOF

echo "=== Creating metadata.json for train ==="
cat > "$CACHE_DIR/train/metadata.json" << 'EOF'
{
  "model_name": "Qwen/Qwen3.5-0.8B",
  "model_revision": null,
  "start_layer": 8,
  "end_layer": 11,
  "span_depth": 4,
  "seq_len": 128,
  "store_logits": false,
  "num_samples": 19000
}
EOF

# Remove old metadata from root
rm -f "$CACHE_DIR/metadata.json"

echo ""
echo "=== Verifying split ==="
echo "Val shards: $(ls -1 $CACHE_DIR/val/*.safetensors 2>/dev/null | wc -l)"
echo "Train shards: $(ls -1 $CACHE_DIR/train/*.safetensors 2>/dev/null | wc -l)"
echo ""
echo "=== Done! Cache split complete ==="
echo "Train: $CACHE_DIR/train/ (19000 samples)"
echo "Val: $CACHE_DIR/val/ (1000 samples)"
