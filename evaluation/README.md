This repository includes implementation details of the draft

## Evaluation.

Evaluating an unlearned model is particularly challenging, as direct comparisons of model parameters often fail to capture practical behavior. To address this, recent works propose benchmarks like TOFU and RWKU that involve prompting the model and assessing its responses to evaluate the effectiveness of unlearning methods. Inspired by those benchmarks,  to evaluate a model's performance on facts in $\mathcal{F}$, we iterate over them, and for a fact $(s, r, o) \in \mathcal{F}$, we obtain the model's prediction about the relation $r$ of entity $s$.

More concretely, to extract model's prediction for the pair $(s, r)$, we provide $5$ in-context examples with other entities, and this particular relation $r$, along with their corresponding values to teach the model to generate the value in response [^1].

We refer to Appendix E for more details of in-context learning for extracting model prediction. We then query the $(s, r)$ of interest and obtain models generated value $o$. We analyze both the model generation and logits probabilities. Model generation reflects the final answer predicted by the model, while logits reflect the probability distr

<br>

#### Generating Models' Prediction for a pair $(s, r)$ and Use ChatGPT as Judge

The files [`eval_clean_model_in_context.py`](/evaluation/eval_clean_model_in_context.py), [`eval_corrupted_model_in_context.py`](/evaluation/eval_corrupted_model_in_context.py), [`eval_unlearned_model_in_context.py`](/evaluation/eval_unlearned_model_in_context.py) utilize the [`eval_utils.py`](/evaluation/eval_utils.py) that obtains models' prediction (clean, corrupted, unlearned) and save them in corresponding files. Then, models' predictions can be evaluated with ChatGPT [`judge.py`](chat_gpt_eval/judge.py) and the judge's output can be found in [`judge_outputs/`](/evaluation/chat_gpt_eval/judge_outputs/).

<br>

#### Logits Analysis

The files [`mcqa_clean_model.py`](/evaluation/mcqa_eval/mcqa_clean_model.py), [`mcqa_corrupted_model.py`](/evaluation/mcqa_eval/mcqa_corrupted_model.py), [`mcqa_unlearned_model.py`](/evaluation/mcqa_eval/mcqa_unlearned_model.py) utilize the [`mcqa_utils.py`](/evaluation/mcqa_eval/mcqa_utils.py) that obtains models' logits information for different categroy of outputs (clean, corrupted, random).

[^1]: This is required as we are working with pretrained base models, not chat-base instruction tuned ones. Context helps the model generating desired outputs, simplifying evaluation.