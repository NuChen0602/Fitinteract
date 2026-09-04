
运行仓库中的全部样本：

```powershell
python run_mvp.py --task all --model minicpm --data-root ".\data"
```

只运行一个任务或样本：

```powershell
python run_mvp.py --task bad_good --model minicpm --sample bad_good_001 --data-root ".\data"
```

## Qwen


Qwen 每 0.4 秒提交当前输入，并调用一次 `response.create` 请求文本响应。

运行仓库中的全部样本：

```powershell
python run_mvp.py --task all --model qwen --data-root ".\data"
```

只运行一个任务或样本：

```powershell
python run_mvp.py --task bad_good --model qwen --sample bad_good_001 --data-root ".\data"
```

## reps_001 示例

如果完整数据集位于 `I:\MVP demo`：

```powershell
python run_mvp.py --task reps --model minicpm --sample reps_001 --data-root "I:\MVP demo"
python run_mvp.py --task reps --model qwen --sample reps_001 --data-root "I:\MVP demo"
```

结果默认写入 `results\<task>\`，汇总文件为 `results\summary.csv`。

可用任务：`reps`、`bad_good`、`accident`、`multiplayer`、`all`。
