# Belle II public likelihood for B+ -> K+ nu nu-bar

Source: Belle II Collaboration, "Model-agnostic likelihood for the reinterpretation of the
B+ -> K+ nu nu-bar measurement at Belle II", arXiv:2507.12393, Phys. Rev. D 112, 092016 (2025).

- HEPData record: https://www.hepdata.net/record/ins2947386 (doi:10.17182/hepdata.166082),
  downloaded in YAML format with its additional resources on 2026-11-16. The record version
  is not stated in the files.
- License: CC0.

Files used by lights (`lights.load_data.BelleII_public`):

| File | Contents |
| --- | --- |
| `combined_likelihood.json` | pyhf (HistFactory JSON) workspace of the ITA and HTA analyses |
| `number_density_ita_regA.csv` | joint number density of ITA region A (eta(BDT2) in [0.94, 1]): 9 likelihood bins x 101 true-q^2 bins |
| `number_density_ita_regB.csv` | the same for ITA region B (eta(BDT2) in [0.92, 0.94]): 3 x 101 |
| `number_density_hta.csv` | the same for the HTA: 6 x 101 (not used yet) |

The other files (`submission.yaml`, the `joint_number_density_*.yaml` tables and the images)
are the rest of the record. The true-q^2 bins are [-1, 0] (events outside the kinematic range)
followed by 100 equal bins from 0 to 22.885 GeV^2. The SM spectrum the number densities were
generated with is not part of the record; it is implemented in
`lights.load_data.belle2_null_spectrum` from Belle II's examples repository
(https://github.com/lorenzennio/knunu-reinterpretation-mini, `scripts/utils.py`).
