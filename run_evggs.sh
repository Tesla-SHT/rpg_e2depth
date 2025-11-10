#!/bin/bash

# 配置参数
MODEL_PATH="pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar"
DATA_ROOT="/run/user/1001/gvfs/sftp:host=login.cvgl.lab,port=22332/datasets/feed_forward_event/Tartanair_tmp/indoor"
OUTPUT_FOLDER="./output"
START_IDX=1
STOP_IDX=100

# 检查数据根目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# 获取所有场景目录
echo "=== Scanning for scenes in $DATA_ROOT ==="
SCENES=()
for dir in "$DATA_ROOT"/*/; do
    if [ -d "$dir" ]; then
        scene=$(basename "$dir")
        # 检查是否有序列1的数据
        if [ -d "$dir/1" ]; then
            SCENES+=("$scene")
        fi
    fi
done

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No valid scenes found in $DATA_ROOT"
    exit 1
fi

echo "Found ${#SCENES[@]} scenes:"
for scene in "${SCENES[@]}"; do
    echo "  - $scene"
done
echo ""

# 询问是否继续
read -p "Process all ${#SCENES[@]} scenes? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# 计数器
SUCCESS=0
FAILED=0
FAILED_SCENES=()

# 遍历所有场景
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    current=$((i + 1))
    total=${#SCENES[@]}
    
    echo ""
    echo "=========================================="
    echo "[$current/$total] Processing scene: $scene"
    echo "=========================================="
    
    python run_depth.py \
        -c "$MODEL_PATH" \
        -i "$DATA_ROOT" \
        --dataset_type evggs \
        --dataset_name "Tartanair/$scene" \
        --scene_name "$scene" \
        --start_idx $START_IDX \
        --stop_idx $STOP_IDX \
        --output_folder "$OUTPUT_FOLDER" \
        --show_event \
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $scene"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ Failed to process $scene"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$scene")
    fi
done

# 打印总结
echo ""
echo "=========================================="
echo "Processing Summary"
echo "=========================================="
echo "Total scenes: ${#SCENES[@]}"
echo "Successfully processed: $SUCCESS"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed scenes:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "  - $scene"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_FOLDER"
echo "Each scene has its own subfolder:"
echo "  $OUTPUT_FOLDER/<scene_name>/reconstruction/frames/"