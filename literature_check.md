# Motivation & Novelty Framing: Parameter-Level Knowledge/Reasoning Isolation for Financial LLMs

**Prepared for:** Mentor review
**Date:** September 2026
**Prior work anchors:** FinEval-KR (Dou et al., 2025, arXiv:2506.21591, FinNLP@EMNLP 2025) + Math Neurosurgery / MathNeuro (Christ et al., 2024, arXiv:2410.16930, ACL 2025)

---

## 1. What each paper actually does

**FinEval-KR** operates at the *behavioral/output* level. It's an evaluation framework, not an interpretability method. It:
- Decouples "knowledge" from "reasoning" via a three-stage protocol: (1) free-form solve, (2) re-solve with knowledge points supplied, (3) diagnose whether the original failure was a knowledge gap or a reasoning gap.
- Produces four metrics (Accuracy, Knowledge Score, Reasoning Score, and five Bloom's-taxonomy Cognitive Scores).
- Ships a 22-subfield Chinese financial reasoning dataset.
- Its finding: knowledge application is a bottleneck even for top models, and specialized finance models lag general LLMs — but the method never opens the model up. It infers "knowledge vs. reasoning" from prompting/re-answering behavior, not from weights or activations.

**MathNeuro** operates at the *parameter* level. It's a mechanistic/interpretability method, not an evaluation framework. It:
- Uses a Wanda-style, forward-pass-only importance score (weights × activations) to find math-important parameters, then subtracts out parameters also important for general language, leaving a small math-specific subset (~1.5–1.8% of parameters).
- Validates the isolation causally: pruning the identified set deletes math ability while leaving general language intact; scaling it up improves GSM8K/MATH performance 4–17%/5–35% without further training.
- Domain: math only. No knowledge/reasoning decomposition — it isolates "math skill" as a single monolithic capability, not knowledge vs. reasoning within math.

**The key structural difference:** FinEval-KR decouples two *capabilities* (knowledge, reasoning) but stays at the behavioral surface. MathNeuro decouples a *domain skill* from general ability but doesn't further split that skill into knowledge vs. reasoning, and doesn't touch finance.

## 2. Literature check: has this combination been done?

Searched for direct precedent (FinEval-KR + MathNeuro combination, financial neuron/parameter isolation, domain-specific skill localization applied to finance, and knowledge/reasoning decoupling at the parameter level). Findings:

- **No paper combines these two specific works.** No citation trail links Dou et al. (2025) and Christ et al. (2024); they sit in disjoint literatures (financial NLP benchmarking vs. LLM interpretability/pruning).
- **Adjacent work exists and needs to be addressed head-on, not ignored:**
  - *Decoupling Knowledge and Reasoning in LLMs* (Yang, Gao, Wu — Tsinghua, arXiv:2507.18178, AAAI) does knowledge/reasoning decoupling across domains (including some quantitative ones) using a *behavioral* dual-system-prompting method (fast/slow thinking), not parameter-level isolation. This is the closest existing bridge, but it's still surface-level and not finance-specific, and its layer-level claim ("knowledge in lower layers, reasoning in higher layers") is coarse — it doesn't produce an actionable, prunable/scalable parameter set the way MathNeuro does.
  - General-purpose domain/task neuron localization work exists (e.g., Cus-Prun for domain-specific pruning, LLM-Sieve for task-specific pruning, skill-neuron papers cited in MathNeuro's related work) — but none target finance, and none cross-reference a knowledge/reasoning split.
  - Finance-specific LLM work (BizFinBench, FinanceQA, Trading-R1, Market-Bench, CFinBench) is entirely benchmark- or RL-training-focused; none does mechanistic parameter analysis.
- **Conclusion: the gap is genuine.** Nobody has (a) applied MathNeuro-style forward-pass parameter isolation to financial reasoning, or (b) used a FinEval-KR-style knowledge/reasoning decomposition as the *target signal* for parameter-level isolation in any domain, finance or otherwise. The intersection of "domain = finance," "granularity = parameter/mechanistic," and "decomposition = knowledge vs. reasoning" is empty in current literature as of this check.

## 3. Motivation

Financial LLM failures matter differently depending on their source. A model that fails because it lacks a financial fact (e.g., the formula for CAGR) needs a knowledge fix (retrieval augmentation, targeted fine-tuning on facts). A model that fails because it can't chain multi-step quantitative reasoning under financial framing needs a reasoning fix (different intervention entirely — e.g., the kind of parameter scaling MathNeuro demonstrates for math). FinEval-KR shows this distinction is real and diagnostically useful in finance, but it can only tell you *that* a failure is knowledge- or reasoning-driven after the fact, via re-prompting — it gives no lever to intervene on the model itself. MathNeuro shows that a reasoning-like skill (math) can be causally isolated to a small, targeted parameter subset and directly manipulated (deleted or amplified) — but it has no notion of knowledge vs. reasoning within that skill, and has never been pointed at finance.

Combining them asks a question neither paper asks alone: **within a financial LLM's parameters, can we isolate the subset responsible for financial-reasoning-as-distinct-from-financial-knowledge, the same way MathNeuro isolates math from general language — and does manipulating that subset produce the same clean, causal, non-destructive effect that MathNeuro demonstrates for math?**

## 4. Novelty framing

1. **Domain novelty:** First application of forward-pass, Wanda-style parameter importance isolation to the financial reasoning domain. MathNeuro's own limitations section frames math as a test case for a generalizable method; finance is a natural and unaddressed next domain — arguably harder, since financial reasoning is more entangled with retrieved factual/regulatory knowledge than math is.
2. **Granularity novelty:** First attempt to take a knowledge/reasoning decomposition (currently only demonstrated behaviorally, via FinEval-KR's re-prompting protocol or the Tsinghua dual-system paper's fast/slow-thinking prompting) and instantiate it as a *parameter-level* split, rather than an output-level diagnosis.
3. **Methodological contribution, not just application:** MathNeuro isolates one skill by subtracting out general-language-important parameters. Extending it to isolate *two* nested things at once (finance-specific parameters, then within those, reasoning-specific vs. knowledge-specific) requires a genuine methodological extension — likely a two-stage filtering process — not a drop-in reapplication.
4. **Falsifiable, causal validation path already exists as a template:** MathNeuro's pruning/scaling validation protocol (does removing the identified parameters delete only the targeted skill; does scaling only it improve targeted-skill performance without collateral damage) transfers directly as the validation methodology, and FinEval-KR's KS/RS metrics and dataset give a ready-made, domain-appropriate evaluation signal to test against — reducing the risk this becomes an untestable interpretability claim.

## 5. Anticipated pushback (worth pre-empting with mentor)

- **"Isn't this just gluing two unrelated papers together?"** The response is #3 above — the combination is not just running MathNeuro on FinEval-KR's dataset; the interesting technical problem is whether financial-reasoning parameters can be *further* decomposed into knowledge- vs. reasoning-specific subsets, which requires extending MathNeuro's filtering logic, not just swapping the domain corpus.
- **"Financial knowledge and reasoning may be too entangled to separate at the parameter level, unlike math."** This is a legitimate risk, not a reason to avoid the project — it's arguably the central research question, and a negative result (entanglement is too high to isolate cleanly) is itself informative given FinEval-KR's finding that knowledge application is the dominant bottleneck for top models.
- **"The FinEval-KR dataset is Chinese-language and judge-model-dependent."** Worth flagging: FinEval-KR's own limitations section notes judge-model dependence (Qwen2.5-72B-Instruct) and that later papers noted their released statistics reflect single runs. Any reuse of their KS/RS metrics as ground truth for parameter-level validation should account for judge-model noise, and language choice (Chinese-only) may constrain which base LLMs are usable for the parameter-isolation side.

## 6. Suggested framing for the actual research question (for discussion)

> Can forward-pass parameter-importance isolation (à la MathNeuro) be extended to decompose financial reasoning ability in LLMs into knowledge-specific and reasoning-specific parameter subsets, using FinEval-KR's knowledge/reasoning decoupling protocol as the behavioral ground truth for validation?

This is intentionally narrower than "combine FinEval-KR and MathNeuro" — it names the mechanism (extended parameter filtering), the domain (finance), and the validation signal (FinEval-KR's KS/RS), which is what will need to be defended in a proposal.
