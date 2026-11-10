# FHM-100 自动研究扫描（stock1）

GitHub Actions 工作日北京时间 16:00 自动运行，也支持手动运行。

## 当前功能
- AkShare A股快照及日线数据；最多扫描成交额靠前的350只非ST沪深主板/创业板股票。
- 基础流动性、5/20日趋势、CLV、回撤诊断。
- 输出 Markdown 报告、CSV观察名单、数据异常记录。
- 可选 AI 风险点评；可选 PushPlus 手机推送。
- Actions Artifact 保存每次运行结果。

## 重要边界
当前是数据接入与技术观察框架，**不是完整正式FHM-100信号引擎**。行业相对强度、市场/行业滚动Beta残差、完整涨跌停事件拆分、经样本外校准的M6收益预测、M8组合风险和M9回测尚未完成。因此不伪造A/B/C正式分数、P_up或预期收益，不把观察名单称为买入名单。原模型80分门槛与风险硬约束不被擅自放宽。

完整规则见 `docs/validated_FHM-100_V2.2-R2.1_完整模型.md`。

## 配置 Secrets
仓库 Settings → Secrets and variables → Actions → New repository secret：

| Secret | 必需 | 说明 |
|---|---|---|
| PUSHPLUS_TOKEN | 推送需要 | PushPlus Token |
| AI_API_KEY | 否 | OpenAI-compatible API Key |
| AI_API_BASE_URL | 否 | 默认 https://api.openai.com/v1 |
| AI_MODEL | 否 | 默认 gpt-4o-mini |

不配置AI仍可扫描；不配置PushPlus则只保存Actions报告。

## 运行和查看
Actions → FHM-100 Daily Research → Run workflow。完成后在运行详情的 Artifacts 下载报告包，包含 report.md、watchlist.csv、data_errors.csv、run_status.json。

不保存密钥到代码，不自动下单，不承诺收益。数据缺失时明确降级。

## R2.1 工程化进度

已新增：
- `src/model/r21_core.py`：A/B/C 原始权重常量、缺失因子拒绝评分、决策时点数据过滤、市场状态基础判定、组合风险预算、RISK_OFF/关键数据缺失时禁止开仓、日线止盈止损双触发保守处理、退出状态与部分成交记账。
- `tests/test_r21_core.py`：对应回归测试。
- 依赖增加 `tabulate`（修复报告 `to_markdown()` 的运行依赖缺失）和 `pytest`。

**重要：这些是 R2.1 安全基础组件，尚未接入每日扫描器，也不代表正式 A/B/C 评分或完整回测已经完成。** 目前的 AkShare 扫描仍只是观察名单；在公司行为账本、点时数据和执行撮合完成前，不得将其当作交易信号。
