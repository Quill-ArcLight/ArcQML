# 示例数据来源

## German Credit

`german_credit.csv` 对应 Open Data LMU 的 **Kreditscoring zur Klassifikation von Kreditnehmern**（2010）。

- 来源页：https://data.ub.uni-muenchen.de/23/
- DOI：https://doi.org/10.5282/ubm/data.23
- 原始数据：https://data.ub.uni-muenchen.de/23/2/kredit.asc
- 变量说明：https://data.ub.uni-muenchen.de/23/1/DETAILS.html
- 数据许可：来源页明确标注 **Open Data Commons PDDL 1.0**，正文见 [LICENSE-PDDL](LICENSE-PDDL)。数据保留该许可下的权利，不受 ArcQML 自有代码非商业条件限制。

2026-09-15 核对：直接下载文件与用户提供的 `kredit.asc.txt` 字节完全一致；与仓库 CSV 的 1,000 行 × 21 列数值逐项一致，行列顺序也一致。转换只需替换表头并将空格分隔写为逗号分隔，不需要改值、重排、筛选或标签重编码。下面描述的是已验证可重现的转换，不主张掌握最初转换时所用的软件或时间。

原始 `kredit.asc` SHA-256：`adeef0d85614229d15295903149db6d0aa5d838df50f69e068c5a5b8dc99bd38`。
本地 CSV SHA-256：`06b9abc60f2b500b2b20d49ec751938b8cd5db29a7ff576513ae95156f5d9d5e`。

| LMU 原列名 | 仓库 CSV 列名 |
|---|---|
| `kredit` | Creditability |
| `laufkont` | Account Balance |
| `laufzeit` | Duration of Credit (month) |
| `moral` | Payment Status of Previous Credit |
| `verw` | Purpose |
| `hoehe` | Credit Amount |
| `sparkont` | Value Savings/Stocks |
| `beszeit` | Length of current employment |
| `rate` | Instalment per cent |
| `famges` | Sex & Marital Status |
| `buerge` | Guarantors |
| `wohnzeit` | Duration in Current address |
| `verm` | Most valuable available asset |
| `alter` | Age (years) |
| `weitkred` | Concurrent Credits |
| `wohn` | Type of apartment |
| `bishkred` | No of Credits at this Bank |
| `beruf` | Occupation |
| `pers` | No of dependents |
| `telef` | Telephone |
| `gastarb` | Foreign Worker |

`kredit` 的 1 表示按约还款，0 表示未按约还款。类别编码应以 LMU 变量说明为准。
`Concurrent Credits` 是本仓库保留的列名，对应 `weitkred`（其他信贷）；不能把它的类别值直接理解为贷款笔数。

与另一个 UCI German Credit 发布版本比较时，存在 53 行、56 个字段差异。这是两个来源版本的比较结果，不是本仓库相对 LMU 来源的修改记录，也不是替换仓库数据的理由。

## H₂ Hamiltonian

`h2_hamiltonian.json` 由收到的 [生成脚本](../python/generate_molecule_hamiltonians.py) 于 2026-09-15 重新计算验证：PennyLane 0.45.1、NumPy 2.4.6、SciPy 1.17.1，后端 `dhf`，STO-3G、0.70 Å、2 电子/2 轨道、Jordan–Wigner 映射。

全部 15 个 Pauli 项及系数与仓库数据完全相同，最大系数差为 0；仅 JSON 排版不同。详细版本、参数与哈希见 [生成记录](h2-generation-record.json)，完整环境版本见 [requirements-h2.txt](../python/requirements-h2.txt)。该记录是本次成功复现的真实环境，不追溯声称它就是最初制作数据时的环境。

复现：使用 Python 3.11，安装 `python -m pip install -r examples/python/requirements-h2.txt`，执行 `python examples/python/generate_molecule_hamiltonians.py`。脚本会写入 examples/data/h2_hamiltonian.json；如需保留原始排版，请在独立副本中执行。它输出由原项目脚本计算的数值，不把 PennyLane 的代码许可证自动套用到计算结果；第三方 Python 包仍保留各自许可。
