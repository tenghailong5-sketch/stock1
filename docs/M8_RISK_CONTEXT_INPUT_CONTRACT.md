# M8 动态风险上下文输入契约

## 目的

M8 只消费操作者明确提供的账户与交易计划上下文，并调用仓位 sizing。缺少真实输入时必须返回未配置/阻断，不得用默认账户余额、持仓数或风险预算伪造通过状态。

## 输入位置

当前运行器读取 `FHM_RISK_CONTEXT_JSON`（GitHub Actions Secret）或本地风险上下文文件。JSON 根节点必须是对象，键为六位 A 股代码，值为该股票的计划对象。

## 每只股票必需字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `entry_price` | number | 计划入场价，必须大于 0 |
| `stop_price` | number | 止损价，多头必须低于入场价 |
| `quantity` | integer | 计划股数，正整数 |
| `planned_risk` | number | 含费用、滑点及跳空预留的计划风险 |
| `reward_risk` | number | 计划盈亏比 |
| `minimum_reward_risk` | number | 策略要求的最低盈亏比，必须大于 0 |
| `existing_portfolio_risk` | number | 当前组合已承担风险，非负 |
| `existing_position_count` | integer | 当前持仓数量，非负整数 |
| `available_risk_budget` | number | 扣除当前敞口和组合约束后的动态剩余风险预算，必须大于 0 |

## 运行时校验

- 市场状态和 M3 风险系数必须有效；M2/M3 不允许新开仓时，M8 不能覆盖前序阻断。
- 计划风险不得低估入场价至止损价的毛风险；仓位 sizing 还会计入费用估算和整手约束。
- `available_risk_budget` 已经扣除当前组合敞口，不得再次扣除 `existing_portfolio_risk`。
- 资金余额、当前持仓、组合风险预算和计划价格必须来自真实账户/操作者输入；不得填示例数字后用于实盘。
- 缺少某只候选的风险对象，只阻断该候选的 M8 决策，不得把其他候选的风险上下文复制过来。

## Secret 配置方式

在 GitHub 仓库 Settings → Secrets and variables → Actions 中新增 Secret：
- 名称：`FHM_RISK_CONTEXT_JSON`
- 值：按上述结构提供 JSON 对象；只放入实际账户数据和实际计划，不要把账户密码、券商登录凭据或 API 私钥放入此字段。

若尚未准备真实账户数据，保持 Secret 不配置。报告应显示 `NOT_CONFIGURED`，所有正式下单授权保持关闭。这不影响全市场扫描、研究排序和 15 分钟结构观察。

## 安全边界

FHM-100 当前只生成研究报告和风险测算，不连接券商、不提交订单。M8 放行也不等于自动交易授权。
