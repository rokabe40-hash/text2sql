# AlexNet 实验（本地数据版）

本仓库包含 `AlexNet` 在本地 `train` 文件夹上的图像分类实验代码，按 **80% 训练 / 20% 测试** 自动划分数据集。

## 目录组织
请将数据按 `ImageFolder` 结构放在工作目录下的 `train` 文件夹，例如：

```text
train/
  airplane/
  automobile/
  bird/
  cat/
  deer/
  dog/
  frog/
  horse/
  ship/
  truck/
```

## 运行方式
```bash
python alexnet_cifar10_local.py \
  --data-dir /home/runner/work/-/-/train \
  --epochs 10 \
  --batch-size 128 \
  --hidden-list 256 512 1024
```

## 输出结果
默认保存在 `outputs/`：
- `alexnet_hidden_width_curves.png`：训练损失/训练精度/测试精度曲线
- `alexnet_hidden_width_results.json`：完整实验配置与结果汇总
