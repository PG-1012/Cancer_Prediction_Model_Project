# Change log

Everything changed while turning this from a coursework submission into a
showcase repo, on 2026-09-01.

**Nothing was deleted.** Files that are now redundant are flagged below
rather than removed.

---

## Files flagged as redundant (kept, not deleted)

| file | why | suggested action |
|---|---|---|
| `bcancer_data (1).csv` | Copied to `data/breast_cancer_wisconsin.csv` under a clean name. The two files are byte-identical; the space and `(1)` in the original name make it awkward to reference from code. | Safe to delete once you've confirmed nothing external links to it. |
| `Model_1.7.ipynb` | Superseded by `src/cancer_model/`, but it is the original submission and the Colab badge points at it. | **Keep.** It is referenced in the README as the original notebook. |

Nothing else was made redundant.

---

## Added

### `src/cancer_model/` — the analysis as a runnable package

| file | what it does |
|---|---|
| `data.py` | Loads the CSV and **validates it** — rejects missing values, non-binary labels, and features outside the 1–10 range. The canonical Wisconsin dataset encodes 16 unknown `Bare.nuclei` readings as `?`; this fails loudly rather than letting them through. |
| `models.py` | Five candidates behind one interface: majority-class baseline, logistic regression, random forest, and the original MLP both as-written and with scaling. Scaling happens **inside each pipeline**, so it is fitted per fold rather than on the whole dataset. |
| `evaluate.py` | Repeated stratified k-fold, out-of-fold predictions, and threshold selection for a target recall. |
| `plots.py` | Seven figures. Red consistently means the costly error. |
| `run.py` | Runs everything and writes `docs/results.json`. Every number in the README comes from here. |

### Other

- `tests/test_cancer_model.py` — 14 tests
- `.github/workflows/ci.yml` — CI on every push
- `requirements.txt`, `LICENSE` (MIT), `.gitignore`
- `figures/` — seven generated figures
- `docs/results.json` — machine-readable results

---

## Changed

### `README.md` — rewritten

The previous description read: *"Machine learning model that tries to
determine if a person is at risk of cancer based on different biological
attributes."*

That overstates what the model does, in a way a technical reader would
notice. The inputs are nine cytological attributes scored by a pathologist
**from a fine-needle aspirate that has already been taken**. They are not
patient risk factors — no age, family history or genetics. The model
classifies an existing biopsy sample as benign or malignant; it cannot tell a
person whether they are likely to develop cancer.

The README now says that plainly, and adds a limits section covering sample
size, single-source data, subjective inputs, and the fact that this is not a
clinically validated device.

---

## Findings that changed the story

These came out of the re-analysis and are now the substance of the README.

**1. The reported 96.3% was a favourable split.** It came from one 80/20
split with `random_state=42`. Cross-validated over 50 folds, the same model
averages **94.1% ± 2.4**. Not wrong — just not repeatable, and with 108 test
cases one misclassification moves accuracy by a full point.

**2. The neural network does not beat logistic regression.** 96.4% ± 1.8 for
logistic regression versus 96.4% ± 1.7 for the scaled MLP. Error bars overlap
almost entirely. On 540 samples and nine ordinal features, the network buys
nothing for its cost in opacity and tuning.

**3. Missing feature scaling cost about two points.** The original fed raw
1–10 values straight in. Standardising took it from 94.1% to 96.4% and cut
missed malignancies from 3.68 to 1.94 per fold — a bigger gain than any
architecture change.

**4. Threshold choice matters far more than model choice.** At the default
0.5 cutoff, logistic regression misses **12** of 239 malignancies. Tuned to
0.23 it misses **2**, costing 3 extra false alarms. The 0.5 default is an
arbitrary inheritance from the sigmoid, not a clinical decision.

**5. Accuracy was the wrong headline metric.** A false negative tells someone
with a malignancy they are clear; a false positive means another test.
Accuracy weighs those identically. Recall and the operating point are where
the real decision lives.

---

## Profile README — done

`github.com/PG-1012/PG-1012` was created and populated on 2026-09-01. It now
carries a self-hosted terminal-style banner and a language card, both generated
by scripts in that repo's `assets/` and swapped light/dark via `<picture>`.

`docs/profile-README-draft.md` in this repo was the earlier plain draft; it is
superseded and can be deleted whenever you like.

**Still needs doing by hand:** the repo description on GitHub still reads
*"Machine learning model that tries to determine if a person is at risk of
cancer…"*. Changing it needs the API too. Suggested replacement:

> Benign vs malignant classification of breast cytology samples — a study of
> how decision-threshold choice interacts with asymmetric error cost.

---

## Not changed

- `Model_1.7.ipynb` — untouched, exactly as submitted
- `bcancer_data (1).csv` — untouched
- Git history — nothing rewritten
- No other repository was touched
