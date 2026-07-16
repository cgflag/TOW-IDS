# Git 推送记录（明文备忘）

更新日期：2026-07-16

## 问题原因

分支 `tow-ids-repro` 的提交历史里包含超大文件（`.npy` / `.h5` / `.pcap`，单文件可达约 2GB）。  
GitHub 单文件限制为 **100MB**，因此对该分支直接 `git push` 会反复失败。

本地大文件目录（勿推送，已由 `.gitignore` 忽略）：

- `normalized_packet_data/`
- `test_normalized_packet_data/`
- `wavelet_transformed_data/`
- `test_wavelet_transformed_data/`
- `wavelet_experiments/`
- `models/`
- `*.pcap` / `*.h5` / `*.npy`

## 解决办法（已采用）

新建 **orphan 干净分支**（无含历史中的大文件），只提交代码与小体积结果：

| 项 | 值 |
|---|---|
| 干净分支名 | `tow-ids-code` |
| 远程 | `origin` → `https://github.com/cgflag/Tow-IDS.git` |
| 本地原分支（含大数据，勿直接 push） | `tow-ids-repro` |

## 推送到 GitHub 的命令

```powershell
# 确认在干净分支
git checkout tow-ids-code

# 首次推送并设置上游
git push -u origin tow-ids-code
```

之后在该分支上只需：

```powershell
git push
```

## 日常工作建议

1. **日常开发 / 训练**：可继续用 `tow-ids-repro`（本地保留数据与完整历史）。
2. **需要同步到 GitHub**：切到 `tow-ids-code`，把代码改动 cherry-pick / 手动拷贝过来，再 `git push`。
3. **永远不要**把 `.npy` / `.h5` / `.pcap` 加入提交；推送前可用下面命令自检：

```powershell
git ls-files | Select-String -Pattern '\.(npy|h5|pcap)$'
# 应无输出
```

## 远程对照

| remote | URL | 用途 |
|---|---|---|
| `origin` | `https://github.com/cgflag/Tow-IDS.git` | 自己的云端仓库（push 到这里） |
| `upstream` | `https://github.com/LokeshNaganaboina/Tow-IDS.git` | 原作者仓库（一般只 fetch） |
