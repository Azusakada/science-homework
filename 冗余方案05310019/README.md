# 最优方案 05301950

当前最优方案入口：

```bash
python strict_super_pool_oof.py --seeds 2026,2027,2028,2029,2030 --no-resume
```

诊断方式：每个候选概率列使用固定阈值 0.5 做 outer val 评估。

说明：本文件夹只放当前最优方案原样执行所需的代码文件。运行仍需要项目根目录中的 `data/train_labels.csv`、`data/train/*.npy` 和 `.cache_strict_eeg` 特征缓存。
