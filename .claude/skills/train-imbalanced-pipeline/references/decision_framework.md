# Decision framework

The single biggest lever on how an imbalanced-data pipeline should be built
is not the imbalance ratio — it's what the model output is actually *for*.
A model whose probability drives a human decision needs something
completely different from a model that only needs to rank cases, which
needs something different again from a model that must emit a hard label.
Get this wrong and every downstream choice (metric, threshold, whether
resampling is even on the table) is wrong too.

So before writing any training code, establish these decisions. If the
user's request already answers one, don't ask it again — extract the
answer from what they said and move on. Only ask about what's genuinely
missing. Use judgement about which questions actually apply (e.g. question 3
is moot if question 1 didn't select "hard classification decision").

## The questions, and why each one matters

**1. What does the model's output need to do?**
   - *Drive a probability-based decision made by someone else* (a risk
     score, a probability of churn/fraud/disease that a human or downstream
     rule acts on) → probabilities must be **calibrated**. A predicted 0.7
     needs to mean 70%, not just "more likely than a 0.5."
   - *Rank or prioritise cases* (which claims to review first, which leads
     to call first) with no downstream logic that depends on the probability
     value itself → only **discrimination** matters, calibration doesn't.
   - *Emit a hard yes/no label directly consumed by another system* →
     you need a well-chosen **decision threshold**, on top of either of the
     above.

   This is the single question the user is most likely to already have an
   opinion on ("we need calibrated probabilities because underwriters act on
   them" / "we just need the top-K riskiest accounts flagged"). If they
   haven't said, ask directly: *"Does anything downstream treat the
   predicted probability as a real-world likelihood (e.g. 0.7 = 70% chance),
   or do you only need the model to rank/separate the two classes?"*

**2. If a hard classification decision is required, do real misclassification
   costs exist?**
   - If the user can state (or a domain expert can supply) the actual cost
     of a false positive vs a false negative — money lost, clinical risk,
     operational cost — that's a real cost matrix, and the threshold should
     be *derived* from it (see `cost_sensitive_and_resampling.md`).
   - If not, the threshold should be *empirically optimised* for whatever
     classification metric the user cares about (balanced accuracy, F1,
     recall at a precision floor, etc.) — ask which one, since different
     metrics have different optimal thresholds and can even favour
     different models.
   - Using the class imbalance ratio itself as a stand-in cost (e.g. "false
     negatives are 10x worse because the minority class is 10x rarer") is a
     common shortcut but is not the same as a real cost — flag this
     explicitly if the user proposes it, rather than silently treating
     frequency as cost.

**3. Are there constraints that rule out the normal recommendation?**
   Ask only if relevant signals are present (e.g. the user mentions a huge
   dataset, a legacy system, or explicitly asks about SMOTE/undersampling):
   - Is the dataset so large that training time is a real bottleneck?
     (Only scenario where undersampling is defensible — as a speed
     shortcut, not a performance lever.)
   - Is there a downstream system that hard-codes a 0.5 cutoff and cannot
     be changed to use a tuned threshold? (One of the few scenarios where
     oversampling has a narrow justification.)
   - Is the classifier constrained to something weak (e.g. must be a
     decision tree or linear model for interpretability/regulatory
     reasons)? Weak classifiers are the ones most likely to show a real
     (if modest) benefit from oversampling — strong ones essentially never
     do.

**4. What does the data actually look like?**
   Don't just ask — check. Load the data (or use an existing EDA if one's
   available) and look at: dataset size, minority class count, whether the
   features are all continuous (needed for SMOTE to make sense at all), and
   whether there's an existing preprocessing pipeline this needs to slot
   into. If minority-class examples number in the low hundreds or fewer,
   flag that this may be a data-sufficiency problem no modelling choice can
   fix, and that plotting learning curves is worth doing before investing in
   anything more elaborate.

## Defaults when the user has no strong opinion

If asked and the user genuinely doesn't know or doesn't care:
- Default objective: discrimination (ROC-AUC / average precision), it's
  the least commital choice and works as a model-comparison metric
  regardless of what happens next.
- Default model set: a couple of strong ensemble models (random forest and
  a gradient boosting implementation available in the environment) plus
  logistic regression as a fast, often well-calibrated baseline. Don't
  default to a single model family.
- Default resampling stance: **none**. Train on the original class
  distribution. This is not a placeholder to revisit later — per the
  gating logic in `cost_sensitive_and_resampling.md`, resampling is opt-in
  and requires one of the narrow justifications above, not the default
  starting point.

## After the decisions are made

Once the objective, cost/threshold approach, and any constraints are known,
hand off to:
- `metrics_and_calibration.md` for which metric to optimise/evaluate with
  and how to handle calibration if it's required.
- `cost_sensitive_and_resampling.md` for how to turn costs into a
  threshold, and the (narrow) conditions under which resampling belongs in
  the pipeline at all.

Both reference files assume the decisions above have already been made —
they're about *how* to implement a choice, not *whether* to make it.
