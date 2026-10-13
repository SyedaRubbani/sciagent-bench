# Planner comparison at a budget of 300 compute units

Regret of the reported incumbent (task objective units; lower is better). CI = seeded bootstrap 95% interval of the mean. Solved = regret below the task tolerance (reactor 0.005, heat 0.005, mimo 0.05). AUC = mean log10 regret over the budget (anytime performance; lower is better). p = paired Wilcoxon signed-rank test against the task's best planner over seeds, Holm-corrected.

## reactor

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| scripted-mf | 0.002995 | [0.001827, 0.004646] | 0.002445 [0.001034, 0.003435] | 85% (20) | -2.31 | best |
| sh | 0.009885 | [0.007305, 0.01248] | 0.01021 [0.004841, 0.01491] | 35% (20) | -2.02 | 0.002 |
| bo | 0.01866 | [0.01087, 0.02856] | 0.01303 [0.007144, 0.02043] | 15% (20) | -1.18 | 0.00042 |
| random | 0.02765 | [0.01765, 0.03877] | 0.01951 [0.01023, 0.0341] | 15% (20) | -1.15 | 0.0004 |

## heat

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| scripted-mf | 0.01186 | [0.00867, 0.01569] | 0.009957 [0.006436, 0.01341] | 15% (20) | -1.63 | best |
| bo | 0.04121 | [0.02968, 0.0572] | 0.03422 [0.02427, 0.04185] | 0% (20) | -1.03 | 0.00013 |
| sh | 0.04126 | [0.03387, 0.04868] | 0.0396 [0.02725, 0.05201] | 0% (20) | -1.37 | 1.9e-05 |
| random | 0.07004 | [0.05529, 0.08594] | 0.06048 [0.04406, 0.09257] | 0% (20) | -1.02 | 1.1e-05 |

## mimo

| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |
|---|---|---|---|---|---|---|
| bo | 0.006648 | [0.002861, 0.01177] | 0.003294 [0.001017, 0.005541] | 100% (20) | -0.571 | best |
| scripted-mf | 0.0282 | [0.01341, 0.04834] | 0.01133 [0.003201, 0.04388] | 85% (20) | -1.45 | 0.0083 |
| sh | 0.08849 | [0.05777, 0.1244] | 0.0638 [0.04264, 0.1029] | 40% (20) | -1.06 | 7.6e-06 |
| random | 0.4612 | [0.288, 0.6775] | 0.2763 [0.1716, 0.606] | 5% (20) | -0.162 | 5.7e-06 |
