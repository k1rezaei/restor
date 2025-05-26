## Evaluation

Evaluating an unlearned model is particularly challenging, as direct comparisons of model parameters often fail to capture practical behavior. To address this, recent works propose benchmarks like TOFU and RWKU, which involve prompting the model and assessing its responses to evaluate the effectiveness of unlearning methods.

Inspired by these benchmarks, we evaluate a model's performance on facts in $\mathcal{F}$ by iterating over each fact $(s, r, o) \in \mathcal{F}$ and querying the model's prediction for the relation $r$ of entity $s$.

More concretely, to extract the model's prediction for a pair $(s, r)$, we provide 5 in-context examples using other entities and the same relation $r$, along with their corresponding values. This helps guide the model to generate the expected value $o$ in response[^1].

We refer to Appendix E for more details on the in-context learning setup used to extract model predictions. We analyze both the model’s **generations** and its **logits**:
- *Generations* reflect the final answer produced by the model.
- *Logits* reflect the model’s underlying probability distribution before decoding.

---

### Generating Model Predictions for $(s, r)$ and Using ChatGPT as a Judge

The following scripts generate predictions for different model states:

- [`eval_clean_model_in_context.py`](/evaluation/eval_clean_model_in_context.py)
- [`eval_corrupted_model_in_context.py`](/evaluation/eval_corrupted_model_in_context.py)
- [`eval_unlearned_model_in_context.py`](/evaluation/eval_unlearned_model_in_context.py)

These scripts use [`eval_utils.py`](/evaluation/eval_utils.py) to obtain predictions from clean, corrupted, and unlearned models.

Predictions can then be evaluated with ChatGPT using [`judge.py`](chat_gpt_eval/judge.py). The outputs are saved in [`judge_outputs/`](/evaluation/chat_gpt_eval/judge_outputs/).

---

### Logits Analysis

The following scripts extract logits for various output categories (clean, corrupted, random):

- [`mcqa_clean_model.py`](/evaluation/mcqa_eval/mcqa_clean_model.py)
- [`mcqa_corrupted_model.py`](/evaluation/mcqa_eval/mcqa_corrupted_model.py)
- [`mcqa_unlearned_model.py`](/evaluation/mcqa_eval/mcqa_unlearned_model.py)

These rely on [`mcqa_utils.py`](/evaluation/mcqa_eval/mcqa_utils.py).

---

[^1]: This is necessary because we are working with pretrained base models (not instruction-tuned chat models). Providing context helps guide the model to generate the desired outputs, simplifying evaluation.