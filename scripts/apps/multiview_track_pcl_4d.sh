#!/bin/bash
# /multiview_tracker/ 下的全部场景
# 保存路径: scripts/apps/multiview_track_pcl_4d.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash scripts/apps/multiview_track_pcl_4d.sh

Description:
  Batch-run multi-view tracking + triangulation on every scene under the data root.

Defaults (edit in script or override via env vars):
  DATA_ROOT   data/multiview_tracker
  DEVICES     "cuda:0 cuda:1"
  NUM_POINTS  1200
  LOG_DIR     outputs/logs/multiview_track

Tip:
  This script expects training/app dependencies to be installed and OpenCV usable.
  If you see NumPy/OpenCV import errors, install `numpy<2` or upgrade opencv-python.
EOF
    exit 0
fi

# 配置参数
DATA_ROOT="data/multiview_tracker"
DEVICES="cuda:0 cuda:1"
NUM_POINTS=1200
PYTHON_CMD="${PYTHON_CMD:-python -m retracker.apps.multiview_triangulation --views 01 07 13 19 25 31 37 43}"

# 启用 nullglob，防止无匹配时返回字面量
shopt -s nullglob

# 日志配置
LOG_DIR="${LOG_DIR:-outputs/logs/multiview_track}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/multiview_track_${TIMESTAMP}.log"

echo "========================================"
echo "开始处理多视角跟踪点云"
echo "数据根目录: $DATA_ROOT"
echo "当前工作目录: $(pwd)"
echo "日志文件: $LOG_FILE"
echo "========================================"
echo ""

# 检查数据目录
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 数据目录 $DATA_ROOT 不存在 (当前路径: $(pwd))"
    exit 1
fi

# 使用数组存储场景路径（更可靠）
SCENES=("$DATA_ROOT"/*/)
TOTAL=${#SCENES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "错误: 在 $DATA_ROOT 下未找到任何场景目录"
    echo "调试信息:"
    echo "  目录内容: $(ls -la "$DATA_ROOT" 2>&1)"
    echo "  匹配模式: $DATA_ROOT/*/"
    exit 1
fi

echo "发现 $TOTAL 个场景，开始处理..."
echo ""

# 遍历每个场景
COUNT=0
for scene_path in "${SCENES[@]}"; do
    COUNT=$((COUNT + 1))
    
    # 移除尾部的斜杠获取场景名
    scene_name=$(basename "$scene_path")
    
    # 尝试 images 目录（复数）
    image_path="$scene_path/images"

    # 如果不存在，尝试 image 目录（单数，兼容旧数据）
    if [ ! -d "$image_path" ]; then
        image_path="$scene_path/image"
    fi
    
    echo "[$COUNT/$TOTAL] 处理场景: $scene_name"
    echo "  图像路径: $image_path"
    
    # 检查 image 目录是否存在
    if [ ! -d "$image_path" ]; then
        echo "  ⚠️  跳过: 未找到 images 或 image 目录"
        echo "[$COUNT/$TOTAL] $scene_name: SKIPPED (no image dir)" >> "$LOG_FILE"
        continue
    fi
    
    # 执行处理命令
    if $PYTHON_CMD \
        --force_rerun \
        --devices $DEVICES \
        --num_points $NUM_POINTS \
        --data_root "$image_path" 2>&1 | tee -a "$LOG_FILE"; then
        
        echo "  ✓ 完成: $scene_name"
        echo "[$COUNT/$TOTAL] $scene_name: SUCCESS" >> "$LOG_FILE"
    else
        echo "  ✗ 失败: $scene_name (错误码: $?)"
        echo "[$COUNT/$TOTAL] $scene_name: FAILED" >> "$LOG_FILE"
        # 失败继续处理下一项，如需中断请取消下一行注释
        # exit 1
    fi
    
    echo ""
done

echo "========================================"
echo "全部场景处理完成"
echo "总计: $TOTAL, 处理: $COUNT"
echo "日志: $LOG_FILE"
echo "========================================"
