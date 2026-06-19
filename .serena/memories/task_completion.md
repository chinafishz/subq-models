# 任务完成检查

- 翻译/修改完成后：通读 README.md 确认格式无断裂
- 图片引用路径检查：`grep -oP 'figures/[^)]+' README.md | while read f; do test -f "$f" || echo "MISSING: $f"; done`
- Git diff 确认变更范围
- 无 lint/测试要求
