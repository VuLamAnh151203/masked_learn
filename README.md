# CaMuRe: Casual Multmodal Recommendation

download data, --> data/book
link: 

run exp:
python src/main.py -m gloria

## Counterfactual edge analysis

Run exact masked-edge interventions from the trained book checkpoint:

```powershell
python src/counterfactual_edge_analysis.py --number_of_user 100 --gpu_id 0
```

To select the users with the highest baseline Recall@20 first:

```powershell
python src/counterfactual_edge_analysis.py --number_of_user 100 --user_selection recall_desc --gpu_id 0
```

`--number_of_user` is optional; omitting it selects every eligible test user.
Every incident train edge of each selected user is set to zero separately in
the masked branch, and the script recomputes both target-user metrics and exact
full-test metrics. Results are written incrementally under
`counterfactual_results/`. A stopped run can be continued with the same
arguments plus `--resume`.
