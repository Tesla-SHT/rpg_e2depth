#!/bin/bash

# 配置参数
MODEL_PATH="pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar"
DATA_ROOT="/run/user/1000/gvfs/sftp:host=login.cvgl.lab,port=22332/datasets/feed_forward_event/MVSEC_all"
OUTPUT_FOLDER="./output/MVSEC_150_350"
# DATA_ROOT="/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor"
# OUTPUT_FOLDER="./output/TartanAir_fintuned-082_scaledepth"
START_IDX=150
STOP_IDX=350

# 评估相关参数
EVAL_CLIP_DISTANCE=80.0
REG_FACTOR=3.70378
SAVE_PNG_FLAGS="--save_pred_png --save_gt_png"  # 如果不想保存PNG，设为空字符串

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

# 创建汇总文件
SUMMARY_FILE="$OUTPUT_FOLDER/evaluation_summary.txt"
mkdir -p "$OUTPUT_FOLDER"
echo "=== Evaluation Summary ===" > "$SUMMARY_FILE"
echo "Model: $MODEL_PATH" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Eval Clip Distance: $EVAL_CLIP_DISTANCE" >> "$SUMMARY_FILE"
echo "Reg Factor: $REG_FACTOR" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Scene | abs_rel | delta1.25 | delta1.03 | Status" >> "$SUMMARY_FILE"
echo "------|---------|-----------|-----------|-------" >> "$SUMMARY_FILE"

# 遍历所有场景
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    current=$((i + 1))
    total=${#SCENES[@]}
    
    echo ""
    echo "=========================================="
    echo "[$current/$total] Processing scene: $scene"
    echo "=========================================="
    
    # 运行评估并捕获输出
    output=$(python run_depth_evggs.py \
        -c "$MODEL_PATH" \
        -i "$DATA_ROOT" \
        -o "$OUTPUT_FOLDER" \
        --scene_name "$scene" \
        --start_idx $START_IDX \
        --stop_idx $STOP_IDX \
        --eval_clip_distance $EVAL_CLIP_DISTANCE \
        --reg_factor $REG_FACTOR \
        $SAVE_PNG_FLAGS \
        --use_gpu 2>&1)
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $scene"
        SUCCESS=$((SUCCESS + 1))
        
        # 提取评估指标
        abs_rel=$(echo "$output" | grep "abs_rel:" | awk '{print $2}')
        delta125=$(echo "$output" | grep "delta <= 1.25:" | awk '{print $4}')
        delta103=$(echo "$output" | grep "delta <= 1.03:" | awk '{print $4}')
        
        # 写入汇总文件
        printf "%-30s | %7s | %9s | %9s | OK\n" "$scene" "$abs_rel" "$delta125" "$delta103" >> "$SUMMARY_FILE"
        
        # 显示结果
        echo "  abs_rel: $abs_rel"
        echo "  delta1.25: $delta125"
        echo "  delta1.03: $delta103"
    else
        echo "✗ Failed to process $scene"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$scene")
        printf "%-30s | %7s | %9s | %9s | FAILED\n" "$scene" "N/A" "N/A" "N/A" >> "$SUMMARY_FILE"
    fi
done

# 打印总结
echo "" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Processing Summary" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Total scenes: ${#SCENES[@]}" | tee -a "$SUMMARY_FILE"
echo "Successfully processed: $SUCCESS" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"

if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Failed scenes:" | tee -a "$SUMMARY_FILE"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "  - $scene" | tee -a "$SUMMARY_FILE"
    done
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "Results saved to: $OUTPUT_FOLDER" | tee -a "$SUMMARY_FILE"
echo "Summary file: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "Each scene has:" | tee -a "$SUMMARY_FILE"
# echo "  - predictions/: predicted depth maps (.npy)" | tee -a "$SUMMARY_FILE"
# echo "  - ground_truth/: ground truth depth maps (.npy)" | tee -a "$SUMMARY_FILE"
if [[ $SAVE_PNG_FLAGS == *"save_pred_png"* ]]; then
    echo "  - pred_png/: predicted depth visualizations (.png)" | tee -a "$SUMMARY_FILE"
fi
if [[ $SAVE_PNG_FLAGS == *"save_gt_png"* ]]; then
    echo "  - gt_png/: ground truth depth visualizations (.png)" | tee -a "$SUMMARY_FILE"
fi