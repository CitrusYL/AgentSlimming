# Dataset Notices

This file documents only the git-tracked benchmark fixtures under
`data/datasets/`.

General notes:

- Unless otherwise noted, source code in this repository is MIT-licensed, but
  dataset files are not relicensed under MIT.
- The git-tracked dataset fixtures in this repository were inherited from the
  earlier AFlow repository and copied into AgentSlimming as-is.
- AgentSlimming has not independently reconstructed or diffed each tracked
  fixture against the official upstream benchmark release. AFlow may have
  applied subsetting, split changes, schema conversion, or other preprocessing
  before these files were committed.
- Untracked datasets materialized locally by `python -m data.download` are out
  of scope for this file.
- The upstream license references below were checked against public benchmark
  pages, repositories, or dataset cards when this file was written. Downstream
  redistributors should still verify current upstream terms for their own use.

## DROP

- Local files:
  - `data/datasets/drop_validate.jsonl`
  - `data/datasets/drop_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: DROP: A Reading Comprehension Benchmark Requiring
  Discrete Reasoning Over Paragraphs.
- Upstream references:
  - <https://huggingface.co/datasets/ucinlp/drop>
  - <https://aclanthology.org/N19-1246/>
- Upstream license reference: CC BY-SA 4.0.
- Known provenance caveat: the repository files are AFlow-era repository-local
  fixtures rather than the full official upstream dataset. The exact AFlow
  transformation history has not been independently audited by AgentSlimming.
- Redistribution note: preserve upstream attribution and review CC BY-SA 4.0
  obligations before redistributing copied, reformatted, or modified versions
  of these files.

## GSM8K

- Local files:
  - `data/datasets/gsm8k_validate.jsonl`
  - `data/datasets/gsm8k_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: GSM8K (Grade School Math 8K).
- Upstream references:
  - <https://huggingface.co/datasets/openai/gsm8k>
  - <https://github.com/openai/grade-school-math>
- Upstream license reference: MIT.
- Known provenance caveat: the official upstream distribution uses its own
  canonical split structure, while the repository files here are AFlow-era
  repository-local fixtures. The exact AFlow transformation history has not
  been independently audited by AgentSlimming.
- Redistribution note: keep upstream attribution and preserve the MIT license
  notice chain when redistributing inherited copies of these fixtures.

## HotpotQA

- Local files:
  - `data/datasets/hotpotqa_validate.jsonl`
  - `data/datasets/hotpotqa_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering.
- Upstream references:
  - <https://hotpotqa.github.io/>
  - <https://huggingface.co/datasets/hotpotqa/hotpot_qa>
- Upstream license reference: CC BY-SA 4.0.
- Known provenance caveat: the repository files are AFlow-era repository-local
  fixtures and are not guaranteed to match the official upstream files or split
  structure byte-for-byte. The exact AFlow transformation history has not been
  independently audited by AgentSlimming.
- Redistribution note: preserve upstream attribution and review CC BY-SA 4.0
  obligations before redistributing copied, reformatted, or modified versions
  of these files.

## HumanEval

- Local files:
  - `data/datasets/humaneval_validate.jsonl`
  - `data/datasets/humaneval_test.jsonl`
  - `data/datasets/humaneval_public_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: HumanEval: Hand-Written Evaluation Set.
- Upstream references:
  - <https://github.com/openai/human-eval>
  - <https://huggingface.co/datasets/openai/openai_humaneval>
- Upstream license reference: MIT.
- Known provenance caveat: the official HumanEval release centers on a canonical
  benchmark test set, while the repository files here use AFlow-era
  repository-local `validate`, `test`, and `public_test` artifacts. The exact
  AFlow transformation history has not been independently audited by
  AgentSlimming.
- Redistribution note: keep upstream attribution and preserve the MIT license
  notice chain when redistributing inherited copies of these fixtures.

## MATH

- Local files:
  - `data/datasets/math_validate.jsonl`
  - `data/datasets/math_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: MATH (Measuring Mathematical Problem Solving With the
  MATH Dataset).
- Upstream references:
  - <https://github.com/hendrycks/math>
  - <https://huggingface.co/datasets/hendrycks/competition_math>
- Upstream license reference: MIT.
- Known provenance caveat: the repository files are AFlow-era repository-local
  fixtures rather than a claim of byte-identical publication of the full
  official upstream dataset. The exact AFlow transformation history has not
  been independently audited by AgentSlimming.
- Redistribution note: keep upstream attribution and preserve the MIT license
  notice chain when redistributing inherited copies of these fixtures.

## MBPP

- Local files:
  - `data/datasets/mbpp_validate.jsonl`
  - `data/datasets/mbpp_test.jsonl`
  - `data/datasets/mbpp_public_test.jsonl`
- Immediate provenance: inherited from the git-tracked AFlow fixture and copied
  into AgentSlimming as-is.
- Upstream benchmark: MBPP (Mostly Basic Python Problems).
- Upstream references:
  - <https://huggingface.co/datasets/google-research-datasets/mbpp>
  - <https://arxiv.org/abs/2108.07732>
- Upstream license reference: CC BY 4.0.
- Known provenance caveat: the repository files here use AFlow-era
  repository-local `validate`, `test`, and `public_test` artifacts and are not
  guaranteed to match the official upstream split structure byte-for-byte. The
  exact AFlow transformation history has not been independently audited by
  AgentSlimming.
- Redistribution note: preserve upstream attribution and review CC BY 4.0
  obligations before redistributing copied, reformatted, or modified versions
  of these files.
