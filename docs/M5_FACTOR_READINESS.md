# M5 正式评分输入就绪矩阵

> 审计基准：main 上的 `src/model/r21_core.py`、`src/model/scoring.py`、`src/model/pipeline.py`、`src/fhm100.py` 与 `docs/M4_DATA_SOURCE_MAPPING.md`。本矩阵描述当前代码契约和数据准入状态，不替代 FHM-100 完整策略原文。未取得原始模型中更细的子因子公式时，不自行拆分或更改权重。

## 1. 当前执行结论

- `src/model/r21_core.py` 定义 A/B/C 三套权重，权重分别合计 100。
- `scoring.py` 要求每个加权因子都存在且为有限数值；硬门槛独立判定，缺失或失败不能被高分抵消。
- `pipeline.py` 仅接收已经构造好的因子百分位和门槛字段，是评分适配器，不负责从原始行情生成完整因子。M3 的市场环境映射是已定义的独立因子，不等于市场状态枚举本身。
- `src/fhm100.py` 仅在 M3 独立达到 `READY` 时，向 A/C 传入按 R1 映射计算的 `market_environment`（0—100）；其他加权因子和硬门槛仍未完整，因此 M5 输出仍只能作为阻塞诊断，不得解释为正式分数或买入候选。B 的 `industry_market_environment` 不由单一市场分数替代。
- 80 分为最低准入线；NEUTRAL 市场状态由管线提高至 85 分；RISK_OFF 由策略评分拒绝。不得下调门槛或以缺失值填分。

## 2. A/B/C 权重与数据就绪矩阵

状态定义：`BLOCKED` = 当前无可用的完整、合格输入；`PARTIAL` = 只有代理/部分字段，禁止整组因子计分；`NEXT_DAY_ONLY` = 必须等次日数据真实可用；`OBSERVE_ONLY` = 仅可做前瞻观察。

| 策略 | 因子键 | 权重 | 代码含义/当前证据 | 当前状态 | 解锁条件 |
|---|---|---:|---|---|---|
| A | industry_relative_strength | 20 | 行业相对强度；需行业收益与市场收益同步、PIT 对齐 | BLOCKED | 合格行业历史序列、行业归属与 available_time；同日完整横截面 |
| A | stock_residual_strength | 25 | 个股对市场及行业收益的滚动残差强度 | BLOCKED | 历史行业归属、同步收益序列、滚动窗口与缺失规则验证 |
| A | price_structure | 15 | 价格结构复合因子；现有 OHLC 可计算部分原语，但尚未映射成正式复合百分位 | PARTIAL | 明确原始模型子因子公式、公司行为总收益口径、完整横截面 |
| A | trade_quality | 15 | 成交质量；成交额/量比仅为部分观察，不可替代换手率/订单流 | PARTIAL | 定义必需字段和缺失策略，验证换手率与量价口径 |
| A | market_environment | 15 | M3 READY 时按 R1 映射输出 0—100 市场环境分；非 READY 不提供 | PARTIAL | 完整历史市场状态及因子百分位、可回放 PIT 数据 |
| A | catalyst | 10 | 催化因子；AI 点评不是结构化事件源 | UNAVAILABLE | 公告/新闻原文、首次可用时间、事件分类/去重及风险标签 |
| B | event_type | 20 | 事件类型 | UNAVAILABLE | 经过核验的事件池与适用涨跌停规则 |
| B | pre_event_trend | 15 | 事件发生前趋势 | PARTIAL | 事件发生时点、事件前 PIT 行情与正式因子定义 |
| B | event_trade_structure | 20 | 事件交易结构 | UNAVAILABLE | 盘中触板、开板/回封等可验证分钟级或逐笔事件；日线不能替代 |
| B | industry_market_environment | 15 | 行业/市场环境 | PARTIAL | 事件时点可见的行业和市场因子 |
| B | next_day_price_action | 20 | 次日价格行为 | NEXT_DAY_ONLY | 次日数据真实可用后另行运行 `stage=next_day`，不得进入前一日收盘评分 |
| B | risk | 10 | 风险因子 | BLOCKED | 事件风险、流动性、公告风险等门槛和评分输入均有可核验来源 |
| C | industry_strength | 15 | 行业强度 | BLOCKED | 同 A 行业历史 PIT 输入与策略定义 |
| C | pre_pullback_strength | 15 | 回撤前强度 | PARTIAL | 仅用回撤前可见数据计算，补齐行业/残差定义和横截面 |
| C | pullback_reversal_structure | 25 | 回撤/反转结构 | PARTIAL | 明确原模型复合公式，验证交易日序列、停牌和公司行为 |
| C | trade_quality | 20 | 成交质量 | PARTIAL | 同 A 成交质量完整字段与覆盖验证 |
| C | market_environment | 15 | M3 READY 时复用同一 R1 市场环境分；非 READY 不提供 | PARTIAL | 完整 PIT 市场因子，而非仅有市场状态枚举 |
| C | catalyst_risk | 10 | 催化与风险 | UNAVAILABLE | 可追溯事件数据和风险标签 |

## 3. 硬门槛（不得并入权重抵消）

### A 策略
- `industry_top_30pct`：行业排名前 30%。
- `stock_residual_top_20pct`：个股残差强度排名前 20%。
- `amount_ma20_ge_100m`：20 日平均成交额至少 1 亿元。
- `breakout_or_confirmed_retest`：突破或确认回踩。
- `no_high_volume_stall_or_repeated_rejection`：无高量滞涨或反复受阻。
- 可选输入 `next_open_gap_pct` 若提供必须为有效数值，且不得高于 3%。

### B 策略
- 仅允许 `stage=next_day`。
- `event_subtype` 只接受代码枚举：`hit_up_unsealed`、`close_up_sealed`、`hit_down_unsealed`、`close_down_sealed`。
- `event_data_verified`、`liquidity_and_risk_pass`、`announcement_risk_pass` 必须为显式布尔真值。

### C 策略
- `pre_pullback_residual_positive`、`pullback_between_0_5_and_4pct`、`pullback_volume_le_80pct_prior5d`、`stop_or_reversal_confirmed`、`industry_not_weak`、`no_material_negative_event`、`liquidity_and_risk_pass` 必须为显式布尔真值。

## 4. 生产接入的强制验收

1. **原文公式核对**：逐项对照 FHM-100 正式策略原文，确定每个复合因子的子因子、方向、窗口、权重和边界；本矩阵不授权改写模型规则。
2. **PIT 契约**：每个原始输入携带交易日、带时区的实际 `available_time`、来源、版本/内容哈希；决策时点过滤必须 fail closed。
3. **横截面契约**：同一决策时点、同一策略比较集合；覆盖不足时因子百分位留空，不缩小样本、不填零。
4. **公司行为/交易规则**：未经验证的公司行为区间、历史涨跌停规则、停牌与不可成交状态不得进入正式标签或回测。
5. **硬门槛契约**：所有 gate 必须是显式布尔值；未知不等于通过。
6. **阶段隔离**：B 的次日因子只能在次日阶段生成；不得回填收盘时点。
7. **验收测试**：逐策略测试缺一因子、NaN/Inf、布尔值冒充分数、缺 gate、False gate、RISK_OFF、80/85 分边界及未来数据泄漏。
8. **输出约束**：只有完整因子、所有硬门槛通过且达到对应最低分数，才允许标记为合格候选；观察名单与诊断不得显示成买入信号。

## 5. 当前可执行的开发顺序

1. 先完成原始 FHM-100 因子公式与代码键的一对一核对。
2. 优先补齐历史行业成员/行业收益、市场历史因子和总收益口径；这决定 A/C 核心因子能否成立。
3. 再补齐价格结构与成交质量的正式复合定义，不能把当前观察代理直接升格。
4. 催化/事件与分钟级涨跌停事件作为独立数据管线；B 保持次日阶段隔离。
5. 所有输入契约通过后才写入生产 M5 adapter，并以滚动样本外测试验证，不用单次命中率或当前幸存股票回填历史。

**当前结论：M5 尚未具备正式选股条件。** 当前工程最重要的进展不是生成更多“分数”，而是明确每个缺口及其解锁证据，避免伪信号污染后续回测和实盘决策。
