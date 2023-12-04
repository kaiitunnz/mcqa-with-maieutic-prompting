# Evaluation of the baseline prompting methods

The code and data in this directory implement our evaluation of the baseline prompting methods. You can see the examples of run scripts we provide in `./scripts`.

## Configuration

### Direct prompting

- Nucleus sampling with _temperature = 0.3_ and _top_p = 1_.
- _7-shot_ examples for _CommonsenseQA_ and _4-shot_ examples for _ARC-E_ and _ARC-C_.

### Chain-of-Thought prompting

- Nucleus sampling with _temperature = 0.3_ and _top_p = 1_.
- _7-shot_ examples from [Wei et al.](https://doi.org/10.48550/arXiv.2201.11903) for _CommonsenseQA_.
- _4-shot_ examples from [Huang et al.](https://doi.org/10.48550/arXiv.2210.11610) for _ARC-E_ and _ARC-C_.

### Self-consistency prompting

- Nucleus sampling with _temperature = 0.5_ and _top_p = 1_.
- Sample 40 outputs returned by PaLM 2.
- Each output is obtained with Chain-of-Thought prompting with the same few-shot examples stated above.
- The prediction is determined by majority voting with each sample having an equal weight.

_Note that for Direct and Chain-of-Thought prompting, we could not use greedy decoding because of the PaLM API's security policy, which prohibits PaLM 2 from generating harmful responses, which are necessary for some sample questions in the dataset._
