# 复现说明（写给自己，也写给后来者）

论文：[TOW-IDS (IEEE)](https://ieeexplore.ieee.org/document/9947068)  
上游仓库：[LokeshNaganaboina/Tow-IDS](https://github.com/LokeshNaganaboina/Tow-IDS)  
本仓库分支：`tow-ids-code`（仅代码与小体积结果；`.npy` / `.h5` / `.pcap` 不进 Git）

一句话说现状：**流水线能跑通，Table-II 也训完了，但和论文数字对不齐，我自己不认为已经“复现成功”。**

---

## 我改了什么（相对原始 GitHub 版）

原仓库更像 notebook 搬出来的脚本，拿来直接跑会踩坑。我这边主要动了这些：

1. **预处理按论文意图重做**  
   每个目标尺寸 `{32, 60, 116, 228, 452}` 各自成一套数据；按时间顺序连续分组；组内出现攻击包就标 Abnormal。原始实现数据量偏小，分组逻辑也怪。

2. **模型补了残差**  
   原代码是 `Sequential`，论文里说的 residual / 可分离卷积块对不上。现在改成函数式 API：`Block A` 双 SeparableConv，`Block B` 带 shortcut 的残差（重复 5 次），`Block C` 分类头。

3. **Table-II 网格实验入口**  
   ```bash
   python run_pipeline.py --mode table2
   ```
   覆盖 3 种小波配置 × 多尺寸 × 多分解层级，一共 45 组。支持缓存复用和断点续跑；大尺寸会自动降 batch，不然 DirectML 上容易把设备打挂。

4. **纯 Python 可跑**  
   notebook 还在，但不依赖 Jupyter 也能走完整条 train / test / train_all / table2 链路。

这些是工程上实实在在能用的东西。论文指标对齐是另一回事。

---

## 当前实验结果（本机，45/45）

汇总文件：`results/table2_results.json`  
可读表 + 图（由已有 CSV 生成，不重训）：[`results/summary/table2_summary.md`](./results/summary/table2_summary.md)

重新生成命令：

```bash
python plot_table2_results.py
```

| 项 | 数值 |
|---|---|
| 完成组数 | 45 |
| 平均 accuracy | ~0.973 |
| 平均 F(abnormal) | ~0.971 |
| 最好一组 | `one_wavelet_1_subband` + `60×60` + Lv1，acc≈0.993 |
| 最差一组 | `one_wavelet_4_subbands` + `228×228` + Lv4，acc≈0.929 |

数字看起来不差，但**和论文 Table-II 的趋势 / 最优配置对不上**。我也不打算在 README 里装作已经复现。

训练中途还遇到过离谱低分（比如某次 `60×60` + 4-subband 掉到 0.44），后面重跑有改善，说明训练稳定性本身也有问题——这可能是实现细节、数据切分、或硬件后端（DirectML）里的某一个，暂时没钉死。

---

## 不足（诚实清单）

写出来免得以后自己美化记忆：

1. **没对上论文结果**  
   最优配置、各尺寸相对高低、小波 / subband 的对比结论，都和原文不一致。原因可能在预处理、小波通道拼接、标签规则、训练超参、甚至数据划分，目前分不开。

2. **实现细节仍有猜的成分**  
   论文对若干步骤写得偏概括。小波层数、子带怎么叠成输入通道、残差块内部顺序，都是按文字自己推的，不能保证和作者代码一致。

3. **大文件不在仓库里**  
   PCAP、归一化 `.npy`、小波缓存、`.h5` 模型都在本地（见 `.gitignore`）。别人 clone 下来不能直接 eval，得自己准备数据和重跑。这是有意为之——文件大到 GitHub 根本推不上去。

4. **训练环境偏 Windows + DirectML**  
   大尺寸（尤其 452）吃显存，也更容易出设备重置类错误。换 CUDA 机器行为可能不一样，我这边没有完整对照。

5. **对比基线（ResNet / EfficientNet / no-wavelet）**  
   脚本在，但 Table-II 复现精力主要砸在自定义 TOW 模型网格上，基线侧没有同等力度的系统对比报告。

6. **文档和论文数字的对照表还没整理完**  
   现在只有本机 JSON；缺少一张“论文报的数 vs 我跑出来的数”并排表。这是下一步该补的。

---

## 这个仓库值不值得 star

如果你要的是「点开就能拿到论文同款数字」——现在还不行，别被高 accuracy 骗了。

如果你要的是下面这些，可以留个 star，也欢迎开 issue 怼我：

- 想在原仓库基础上，有一套**能实际跑完 Table-II 网格**的代码
- 需要**补了残差、按尺寸独立建数据**的实现参考
- 想找人一起把「差在哪一步」钉死（预处理？小波？训练？）

我自己的目标很简单：搞清楚差在哪，把对不上的地方收窄，而不是把 accuracy 刷漂亮一点就收工。

---

## 快速上手（代码分支）

```bash
git checkout tow-ids-code
pip install -r requirements.txt

# 全流程（需自备 PCAP / 标签）
python run_pipeline.py --mode train
python run_pipeline.py --mode test

# Table-II 网格
python run_pipeline.py --mode table2
```

本地大数据分支是 `tow-ids-repro`，不要直接 push 那个；公开同步用 `tow-ids-code`。

## 数据集下载

IEEE port: TOW-IDS  [TOW-IDS: Automotive Ethernet Intrusion Dataset | IEEE DataPort](https://ieee-dataport.org/documents/tow-ids-automotive-ethernet-intrusion-dataset)