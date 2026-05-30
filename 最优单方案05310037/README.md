# 最优单方案05310037

只保留 `session_mean_accuracy_6runs.csv` 中原序号 8 的方案：

`conv_fusion_z_log_0.20 = 0.80 * SpectralDomainSVC + 0.20 * ConvNet embedding LogisticRegression`

运行：

```bash
python single_best_session_transfer.py
```

默认执行 3 个 seed：`2026,1145,114399`，每个 seed 跑两组纯 sess 迁移：

- `sess1` 训练，`sess2` 测试
- `sess2` 训练，`sess1` 测试

输出在 `diagnostics/` 下。代码只保留该单方案所需训练流程，不包含其它候选模型、OOF 排名、stacker、spec/ERP 或其它融合权重。
