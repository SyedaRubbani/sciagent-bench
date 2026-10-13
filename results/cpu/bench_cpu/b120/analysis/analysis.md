# Planner comparison at a budget of 120 compute units

Regret of the reported incumbent (task objective units; lower is better). CI = seeded bootstrap 95% interval of the mean. Solved = regret below the task tolerance (reactor 0.005, heat 0.005, mimo 0.05). AUC = mean log10 regret over the budget (anytime performance; lower is better). p = paired Wilcoxon signed-rank test against the task's best planner over seeds, Holm-corrected.

## reactor

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| scripted-mf | 0.01022 | [0.005602, 0.01662] | 0.006263 [0.003031, 0.01189] | 30% (20) | -1.77 | best |
| sh | 0.0178 | [0.01232, 0.02424] | 0.01661 [0.008473, 0.02125] | 20% (20) | -1.77 | 0.097 |
| random | 0.1483 | [0.09008, 0.2112] | 0.09937 [0.02266, 0.2106] | 5% (20) | -0.781 | 0.00014 |
| bo | 0.1483 | [0.09008, 0.2112] | 0.09937 [0.02266, 0.2106] | 5% (20) | -0.781 | 0.00014 |

## heat

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| scripted-mf | 0.03706 | [0.02886, 0.04513] | 0.03574 [0.02441, 0.05163] | 5% (20) | -1.32 | best |
| sh | 0.05797 | [0.0431, 0.07439] | 0.04699 [0.02863, 0.07679] | 0% (20) | -1.26 | 0.083 |
| random | 0.1152 | [0.09372, 0.1366] | 0.1119 [0.07436, 0.1636] | 0% (20) | -0.915 | 5.7e-06 |
| bo | 0.1152 | [0.09372, 0.1366] | 0.1119 [0.07436, 0.1636] | 0% (20) | -0.915 | 5.7e-06 |

## mimo

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| scripted-mf | 0.07196 | [0.0449, 0.104] | 0.04765 [0.03716, 0.07912] | 55% (20) | -0.983 | best |
| sh | 0.258 | [0.1666, 0.3664] | 0.1945 [0.1061, 0.2954] | 15% (20) | -0.638 | 0.00071 |
| random | 1.246 | [0.7057, 1.869] | 0.692 [0.1853, 1.695] | 5% (20) | 0.0652 | 8e-05 |
| bo | 1.738 | [0.8468, 2.865] | 0.4158 [0.2401, 2.33] | 5% (20) | 0.0663 | 8e-05 |
