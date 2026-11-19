#!/bin/bash
echo "======================================"
echo "   OpenAI Agent 首次配置向导"
echo "======================================"
echo "请输入你的 OpenAI API Key (粘贴后按回车，屏幕不会显示):"
read -s API_KEY
echo ""

if [ -z "$API_KEY" ]; then
    echo "❌ 错误：Key 不能为空，配置取消。"
    exit 1
fi

echo "请输入使用的模型名称 (直接回车默认使用 gpt-4o-mini):"
read MODEL
MODEL=${MODEL:-gpt-4o-mini}

cat > .env <<ENVEOF
OPENAI_API_KEY=$API_KEY
OPENAI_MODEL=$MODEL
OPENAI_PROXY_URL=http://127.0.0.1:17897
ENVEOF

echo "✅ 配置成功！已保存到 .env 文件中。"
if ! grep -q "^.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo "🔒 已自动将 .env 加入 .gitignore，防止误传 GitHub。"
fi
