# Belle II and BaBar inputs (digitised)

The inputs of the lights Belle II and BaBar likelihoods (`lights.load_data.BelleII` and
`lights.load_data.BaBar`). All of them were digitised from figures of the publications below;
they are therefore approximations of the published results, with a digitisation uncertainty
not included in the likelihoods.

## Belle II, B+ -> K+ nu nu-bar, inclusive tag analysis (ITA)

Belle II Collaboration, "Evidence for B+ -> K+ nu nu-bar decays", arXiv:2311.14647,
Phys. Rev. D 109, 112006 (2024).

| File | Contents | Source |
| --- | --- | --- |
| `obs_inc.txt` | observed events in 20 bins of reconstructed q^2 (1 GeV^2 wide, from -1 to 19 GeV^2), signal region eta(BDT2) > 0.92 | Fig. 17 |
| `bkg_inc_cont.txt`, `bkg_inc_charged.txt`, `bkg_inc_neutral.txt` | expected continuum, B+B- and B0B0-bar backgrounds in the same bins | Fig. 17 |
| `errors_inc.txt` | uncertainties (error bars) of the observed binned rate in the same bins | Fig. 7 |
| `SignalEfficiency.json` | ITA signal efficiency [%] in 2 GeV^2 bins of true q^2 (0-24 GeV^2): value, lower and upper uncertainty | Fig. 6 |
| `SignalEfficiency_HAD.json` | the same for the hadronic tag analysis (HTA); only compared with the ITA in notebook 02 | Fig. 6 |
| `ITALikelihoodPoints_PDF.json` | published ITA profile likelihood of the signal strength, (mu, -2 Delta ln L); used to validate our Belle II likelihood in notebook 02 | Fig. 16, ITA curve |

The q^2 smearing map is from a Belle II PhD thesis:

| File | Contents | Source |
| --- | --- | --- |
| `q2SmearingMapq2genvsq2rec.txt` | migration from generated (true) to reconstructed q^2, 1 GeV^2 bins from -5 to 25 GeV^2 (rows: true q^2, columns: reconstructed q^2) | Fig. 5.1 of the thesis at https://ediss.sub.uni-hamburg.de/handle/ediss/9793 |

## BaBar, B0 -> K*0 nu nu-bar

BaBar Collaboration, "Search for B -> K(*) nu nu-bar and invisible quarkonium decays",
arXiv:1303.7465, Phys. Rev. D 87, 112005 (2013).

| File | Contents | Source |
| --- | --- | --- |
| `RateBaBarB0Ks0_PDF.json` | observed events in 7 bins of q^2 (0.1 mB^2 wide, from 0 to 0.7 mB^2) | Fig. 5 |
| `PeakBackgroundBabarB0Ks0_PDF.json` | expected total background (peaking + combinatorial) in the same bins | Fig. 5 |
| `CombBackgroundBabarB0Ks0_PDF.json` | expected combinatorial background in the same bins | Fig. 5 |
| `EfficiencyB0toKs0BaBar_PDF.json` | signal efficiency per bin (value, lower and upper uncertainty), as read from the figure; `lights.load_data` multiplies it by 8e-4 | Fig. 6 |

The treatment follows the appendix of arXiv:2403.13887. BaBar searched for B0 -> K*0 and B+ -> K*+ with a hadronic tag, combining two final states
each (K+ pi-, K_S pi0; K+ pi0, K_S pi+). Only B0 -> K*0 is used here: BaBar gives no q^2
smearing, and the B+ -> K*+ final states contain neutral particles, which would smear q^2
non-negligibly and asymmetrically. For B0 -> K*0 the K+ pi- final state dominates (its
efficiency is ten times larger), so the smearing is neglected.

The other BaBar inputs are set in `lights/load_data.py` (`_load_BaBar`):

- the number of BB-bar pairs, 471 million (arXiv:1303.7465);
- the ratio of MC to data statistics of the background, 7.6, fitted so that the statistical
  uncertainties reproduce Table IV of arXiv:1303.7465 (as for Belle II);
- the relative systematic uncertainties of the reconstruction efficiency (8.6%), the peaking
  background (2.8%) and the background normalisation (6.6%), from Tables II and III of
  arXiv:1303.7465, taking into account the correlation between the systematic uncertainties
  of the background and the efficiency.
