# M8 风险上下文配置

M8 只接受操作者提供的账户与交易计划上下文。没有真实账户风险数据时，保持 `NOT_CONFIGURED` 并阻断正式信号；不得用示例值、默认仓位或估算账户资金替代。

## 配置方式

在 GitHub 仓库 Settings → Secrets and variables → Actions → New repository secret 中新增：

- Name：`FHM_RISK_CONTEXT_JSON`
- Value：一个按六位股票代码索引的 JSON 对象。

扫描器从该 Secret 读取配置，不需要把账户数据提交到仓库。也可通过 `FHM_RISK_CONTEXT_FILE` 指向运行环境中的私有 JSON 文件。

## 每只股票必需字段

每个股票对象必须提供：

- `entry_price`：已确认的入场参考价；必须来自当次有效的15分钟确认，不得预填猜测价。
- `stop_price`：结构止损参考价；必须有真实价格结构依据。
- `reward_risk`：按入场价、止损价和目标参考计算的风险收益比。
- `minimum_reward_risk`：账户/策略要求的最低风险收益比。
- `existing_portfolio_risk`：当前组合已占用风险金额。
- `existing_position_count`：当前持仓数量，必须为非负整数。
- `available_risk_budget`：当前剩余可用风险预算金额。
- `same_industry_or_corr_gt_06`：是否已有同产业或相关系数大于0.6的集中持仓，必须为布尔值。

资金字段还必须提供以下二者之一：

- `available_cash`：可用现金金额；或
- `quantity`：正整数股数（仅用于推导可用现金上限，实际下单股数仍由 M8 根据风险预算重新计算）。

## 数据约束

1. 只为已进入当次研究候选池的股票配置上下文；其他股票不需要填充。
2. 不得把观察名单价格当作已确认入场价；15分钟确认未通过时，不生成交易计划。
3. 止损价必须低于入场价，且与真实结构对应；不确定时留空并保持阻断。
4. 风险预算、组合风险、现金和持仓数量必须来自真实账户快照，并在每次扫描前更新。
5. 不要提交账户余额、持仓明细或该 JSON 到 Git 仓库；只放在 GitHub Actions Secret。
6. 缺失字段不填零、不补默认值。M8 未完整时，报告只列观察候选，不发正式信号。

## 配置检查

运行后检查 Artifact 中的 `m8_risk_context_audit.json`：

- `status=LOADED` 且 `symbol_count` 大于0：配置已读取，但不代表每只候选均完整。
- `four_layer_decision_audit.json` 中 `m8_risk_context_complete_symbol_count`：逐只完成必需字段校验的数量。
- `m8_sizing_approved`：风险预算计算是否通过；读取成功不等于仓位获批。

如果没有真实账户风险上下文，请不要为了让数量变成非零而填写虚构值。
