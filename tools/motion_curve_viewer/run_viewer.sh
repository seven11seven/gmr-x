#!/bin/bash
# ===========================================================================
# Motion Curve Viewer — 一键启动脚本
#
# 用法:
#   ./run_viewer.sh                  # 默认端口 8000
#   ./run_viewer.sh 9000             # 指定端口
#   ./run_viewer.sh --help           # 查看帮助
#
# 依赖: python3 (内置 http.server 模块)
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PORT="${1:-8000}"

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Motion Curve Viewer — 启动脚本"
    echo ""
    echo "用法:  $0 [PORT]"
    echo ""
    echo "  PORT    监听端口 (默认 8000)"
    echo ""
    echo "示例:"
    echo "  $0                  # 在 http://localhost:8000 启动"
    echo "  $0 9000             # 在 http://localhost:9000 启动"
    echo ""
    echo "启动后在浏览器打开:"
    echo "  http://localhost:${PORT}/GMR/tools/motion_curve_viewer/motion_curve_viewer.html"
    exit 0
fi

# 检查端口是否被占用
if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用，尝试查找可用端口..."
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    echo "   使用端口: $PORT"
fi

URL="http://localhost:${PORT}/GMR/tools/motion_curve_viewer/motion_curve_viewer.html"

echo "=============================================="
echo "  📊 Motion Curve Viewer"
echo "=============================================="
echo "  项目根目录: $PROJECT_ROOT"
echo "  监听端口:   $PORT"
echo ""
echo "  🌐 浏览器打开:"
echo "     $URL"
echo ""
echo "  按 Ctrl+C 停止服务器"
echo "=============================================="
echo ""

# 尝试自动打开浏览器
if command -v xdg-open &>/dev/null; then
    sleep 1 && xdg-open "$URL" 2>/dev/null &
elif command -v open &>/dev/null; then
    sleep 1 && open "$URL" 2>/dev/null &
fi

# 启动 Python HTTP 服务器
cd "$PROJECT_ROOT"
python3 -m http.server "$PORT"
