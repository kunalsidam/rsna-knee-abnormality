# 03. Heuristic Report Labeling

## Why

4,349 studies have a radiology report but no labels. Reports are written by radiologists and state findings in prose ("There is a complete tear of the anterior cruciate ligament..."). Even crude keyword rules recover some signal, and a model trained on many noisy labels can beat one trained on 46 clean ones. That is a hypothesis, not a result; it has not been tested here.

## How it works (`src/rsna_knee/labels.py`)

1. Split the report into sentences.
2. For each target, find sentences containing any of its keywords (whole-word match).
3. Decide the score:

| Situation | Soft label |
|---|---:|
| Keyword appears in at least one sentence with no negation word | **0.8** |
| Keyword appears, but every such sentence has a negation word | **0.3** |
| Keyword never appears | **0.1** |

Negation words: `intact`, `normal`, `unremarkable`, `preserved`, `no`, `without`, `negative for`, `absent`.

### Keyword rules

| Target | Keywords |
|---|---|
| ACL | acl, anterior cruciate ligament |
| MCL | mcl, medial collateral ligament |
| Medial Meniscus | medial meniscus, medial meniscal |
| Lateral Meniscus | lateral meniscus, lateral meniscal |
| Medial OA | medial compartment, medial femorotibial, medial osteophytes |
| Lateral OA | lateral compartment, lateral femorotibial, lateral osteophytes |
| PF OA | patellofemoral, patella, trochlea, chondromalacia |
| Effusion | effusion, fluid |
| Synovitis | synovitis, synovial |
| Baker's | baker's, popliteal cyst |
| Contusion | contusion, bone bruise, marrow edema |
| Fracture | fracture, break |

### Example

> "The anterior cruciate ligament is intact. There is a small joint effusion."

| Target | Score | Reason |
|---|---:|---|
| ACL | 0.3 | keyword present, sentence contains "intact" |
| Effusion | 0.8 | keyword present, no negation in its sentence |
| Fracture | 0.1 | keyword absent |

The notebook's original version would have scored Effusion 0.3 here, because "intact" appears elsewhere in the report. Sentence-level negation fixes that case.

## Known weaknesses

These rules are a starting point, not a trustworthy labeler.

- **Mention is not diagnosis.** "Mild effusion", "small effusion" and "no effusion" differ in meaning; severity and certainty are ignored.
- **Over-broad keywords.** `fluid` also appears in sequence descriptions; `patella` appears in normal descriptions of the patellar tendon; `break` is rare and odd for fractures. These inflate false positives for Effusion, PF OA and Fracture.
- **Weak OA rules.** Osteoarthritis is rarely described with those exact phrases, so recall is probably low.
- **Negation scope.** A sentence like "No ACL tear, but there is a meniscal tear" contains a negation word, so the meniscus finding gets 0.3 instead of 0.8. Dependency-aware negation (for example NegEx) would handle this better.
- **Hard-coded probabilities.** 0.8 / 0.3 / 0.1 are guesses, not calibrated against the 58 gold studies.

## How to check it

The right test is the 58 gold studies: they have both a report and expert labels. Run `generate_heuristic_labels` on them and compare per target (AUC against the true label, or precision and recall at a 0.5 threshold). Tune the keywords and negation rules until that agreement is acceptable. This evaluation has not been done yet and is the first item on the roadmap.

## Alternatives worth trying

- Replace rules with a small text classifier trained on the 58 gold reports.
- Use an LLM to extract structured findings from reports, validated against the gold set.
- Keep heuristic labels at lower training weight than gold labels (for example by weighting the loss).
