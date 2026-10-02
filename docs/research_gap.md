# Research Gap

## Status: ✅ Defined (Evidence-Based)

---

## 1. Literature Landscape Summary

Based on our review of 17+ papers and targeted evidence searches, the C-MAPSS research landscape can be characterized as follows:

### What the literature does well

| Area | Coverage | Key Observation |
|------|----------|-----------------|
| **Supervised RUL prediction** | Extensive | The dominant paradigm — hundreds of papers using CNNs, LSTMs, Transformers |
| **Unsupervised anomaly detection** | Growing | Smaller but active sub-field using AEs, LSTM-AEs, Isolation Forests |
| **Hybrid models** | Active | AE+IF combinations in latent space are well-explored |
| **Cross-condition transfer** | Active | Domain adaptation (FD001→FD002) studied via adversarial methods |
| **Adaptive thresholding** | Active | Data-driven, adaptive, and multi-level thresholds proposed |
| **Early warning metrics** | Emerging | Recent papers include detection lead time and alarm burden |
| **Complexity vs. performance** | Discussed | Qualitative comparisons between IF efficiency and DL accuracy |

### Honest assessment of our preliminary gap

Our initial hypothesis — that early-warning behavior, cross-condition evaluation, false-alarm analysis, and baseline comparisons receive "comparatively less attention" — was **too broad**. Each of these individual aspects has been addressed to some degree in recent literature.

**We do not claim these topics are unstudied.**

---

## 2. The Refined Research Gap

After evidence-based analysis, we identify a more specific and defensible gap:

> **While individual aspects of unsupervised anomaly detection for predictive maintenance (method comparison, early-warning detection, cross-condition robustness, threshold sensitivity, and computational trade-offs) have each been studied in isolation, there is a lack of unified, methodologically rigorous studies that evaluate all of these dimensions together under a single, reproducible experimental framework with strict leakage prevention on the C-MAPSS benchmark.**

### Specifically, existing work tends to:

1. **Propose novel architectures without strong baselines.** Many papers introduce a new deep learning method and compare it only against other deep methods — or against a weak baseline. Few studies systematically include a well-tuned statistical baseline, Isolation Forest, FC-Autoencoder, AND LSTM-Autoencoder side-by-side under identical conditions.

2. **Report aggregate metrics without engine-level analysis.** Most papers report dataset-wide F1 or AUC, but do not analyze per-engine detection behavior: Which engines were detected early? Which were missed? How does detection vary across individual machines?

3. **Lack transparency in the experimental protocol.** Critical details are often missing or unclear:
   - How were train/validation/test splits constructed? (engine-level? random observation-level?)
   - How was normalization applied? (train-only? full dataset?)
   - How was the anomaly threshold selected? (validation set? test set? post-hoc?)
   - Were the same threshold selection rules applied to all compared methods?

4. **Evaluate on single subsets without cross-subset analysis.** Many studies use FD001 alone. Those that include FD002/FD004 often train and test within each subset, rather than examining how a model trained under simple conditions (FD001) degrades under complex conditions (FD002).

5. **Omit statistical rigor.** Results are typically from a single training run. Confidence intervals, repeated experiments with different seeds, and statistical significance tests are rare.

6. **Conflate model novelty with research contribution.** The contribution is often framed as "our new model outperforms X," when the more scientifically valuable question is "under what conditions and at what cost does each method succeed or fail?"

---

## 3. Our Proposed Investigation

Rather than proposing a novel architecture, our contribution is a **systematic empirical study** that provides:

### 3.1 Unified Experimental Framework
A single, reproducible codebase that evaluates Statistical Threshold, Isolation Forest, Autoencoder, and LSTM-Autoencoder under identical:
- Data splits (engine-level, documented engine IDs)
- Normalization (train-only statistics)
- Threshold selection (validation-set only, multiple strategies)
- Evaluation metrics (classification + early-warning + computational)
- Random seeds (5 seeds per experiment, mean ± std reporting)

### 3.2 Multi-Dimensional Evaluation
Going beyond aggregate F1/AUC to include:
- **Per-engine detection analysis** — Which engines are detected? How early?
- **Detection lead time distribution** — Not just mean, but full distribution across engines
- **False alarm characterization** — Under what conditions do false alarms occur?
- **Threshold sensitivity analysis** — How do results change across threshold strategies?

### 3.3 Cross-Condition Degradation Analysis
Systematic experiments examining:
- Performance within a single operating condition (FD001)
- Performance under multiple operating conditions (FD002)
- Performance with multiple fault modes (FD003)
- Performance under combined complexity (FD004)
- Cross-subset evaluation (train FD001, evaluate FD002)

### 3.4 Honest Complexity Assessment
Quantitative analysis of whether added model complexity is justified:
- Parameter count
- Training time
- Inference time
- Performance gain per unit of complexity
- Concrete recommendation: "When is Isolation Forest sufficient?"

### 3.5 Strict Methodological Transparency
Every experimental decision fully documented:
- Explicit leakage prevention verification
- Published engine ID assignments for each split
- Configuration files for every experiment
- Statistical tests for claimed differences

---

## 4. Why This Gap Matters

### For the research community
- Provides a **reproducible baseline** that future papers can build on and compare against
- Demonstrates whether deep learning is justified for this specific task, or whether simpler methods suffice
- Offers methodological template for fair unsupervised AD evaluation

### For practitioners
- Actionable guidance on method selection under different operating conditions
- Realistic false alarm and detection lead time expectations
- Computational budget guidance (CPU vs GPU, training time)

### For our BTech project
- Demonstrates genuine research methodology (evidence-driven, not metric-chasing)
- Produces defensible, honest results
- Suitable for publication as an empirical study / benchmarking paper

---

## 5. What Our Experiments CAN Demonstrate

- Relative performance of 4 unsupervised methods under controlled conditions
- Effect of operating conditions on detection quality
- Relationship between model complexity and performance
- Per-engine detection behavior and early-warning capability
- Threshold sensitivity across methods
- Reproducible results with documented seeds and splits

## 6. What Our Experiments CANNOT Demonstrate

- Superiority of any single method in general (only on C-MAPSS)
- Real-world deployment performance (C-MAPSS is simulated data)
- Scalability to millions of sensors or thousands of machines
- Online/streaming detection capability (we evaluate offline)
- Performance on non-turbofan industrial systems
- Optimal hyperparameter settings (we explore reasonable ranges, not exhaustive search)

---

## 7. Validation Checklist

- [x] At least 15 relevant papers reviewed
- [x] Gap is validated against existing evidence (not assumed)
- [x] Individual gap components honestly assessed against literature
- [x] Refined gap is specific enough to evaluate
- [x] Gap is addressable within our experimental scope (BTech project)
- [x] Gap is meaningful (unified rigorous evaluation is genuinely needed)
- [x] Our proposed experiments can provide evidence about the gap
- [x] Limitations of our investigation are explicitly stated
- [x] Contribution is framed as empirical study, not architectural novelty

---

## 8. Positioning Statement (for the paper)

> This work does not propose a novel model architecture. Instead, it provides a systematic empirical comparison of unsupervised anomaly detection methods for predictive maintenance, evaluating statistical baselines, Isolation Forest, and reconstruction-based deep learning models under a unified experimental framework with strict leakage prevention. Our study analyzes detection performance, early-warning capability, threshold sensitivity, cross-condition robustness, and computational cost across the NASA C-MAPSS benchmark, offering reproducible baselines and evidence-based guidance on method selection for industrial anomaly detection.
