This repository includes implementation details of the draft

<h2 style="text-align: center;">
RESTOR: Knowledge Recovery through Machine Unlearning
</h2>

<br>
<h4 style="text-align: center;">Abstract</h4>

> <small>Large language models trained on web-scale corpora can memorize undesirable datapoints such as incorrect information, copyrighted content or sensitive data. Recently, many machine unlearning methods have emerged that aim to `erase' these datapoints from trained models. In this work, we propose the RESTOR framework for machine unlearning based on the following dimensions: (1) a task setting that focuses on real-world factual knowledge, (2) a variety of corruption scenarios that emulate different kinds of datapoints that might need to be unlearned, and (3) evaluation metrics that emphasize not just forgetting undesirable knowledge, but also recovering the model's original state before encountering those datapoints, or *restorative unlearning*. RESTOR helps uncover several novel insights about popular unlearning algorithms, and the mechanisms through which they operate-- for instance, identifying that some algorithms merely emphasize forgetting the knowledge to be unlearned, rather than performing restorative unlearning, and that localizing problematic segments within corrupted datasets before unlearning enhances unlearning performance.</small>

<br><br>
<h4 style="text-align: center;">Framework</h4>




![RESTOR Framework](figures/machine_unlearning_teaser.jpg)
<small>*RESTOR framework, including three components: (i) corruption, (ii) unlearning, and (iii) evaluation. Clean model is corrupted by continue pretraining on a set of documents, losing its knowledge on subject entity Nelson Mandela. Unlearning algorithm is then applied to neutralize the effect of corruption documents to ideally obtain a model similar to the clean one. Unlearning may result in <u>forgetting</u> content in corruption dataset, or in <u>ideal</u> case result in <u>recovering</u> the original knowledge.*</small>

<br><br>
<h4 style="text-align: center;">Repo</h4>

This repository includes files for above components:
+ **Corruption** provides implementation details of continual pretraining clean model on corrupted dataset.
+ **Unlearning** provides implementation details on unlearning algorithms discussed in our paper.
+ **Evaluation** provides implementation of (i) evaluating models' outputs with ChatGPT, and (ii) evaluating models' logits layers.
+ **Datasets** provides datasets used in our experiments, including Wikidata and SQuAD.
