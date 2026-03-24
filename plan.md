# Multimodal Embedding Plan

Last updated: 2026-03-24

## Objective

Build a multimodal embedding development stack inside Nexus that is usable for:

- supervised finetuning
- offline embedding inference
- local and benchmark-style evaluation
- MMEB v2 data preparation and conversion

The target project outcome is to train a strong open-backbone multimodal embedder that can compete with and ideally exceed the `Qwen3-VL-Embedding-8B` family on MMEB v2.

## Baseline

- Upstream base repo: `USTCLLM/Nexus`
- Reference repos already inspected locally:
  - `FlagEmbedding`
  - `VLM2Vec`
- First local milestone already committed:
  - commit `8a5d364`
  - message: `Add multimodal embedding pipeline and examples`

## Current Status

Done:

- Added a first multimodal retrieval pipeline to Nexus for training, inference, and evaluation.
- Added example configs and scripts under `examples/multimodal_retrieval/`.
- Confirmed the repo is already under git and local progress is traceable from the original Nexus history.

In progress:

- Hardening the multimodal stack for MMEB v2 readiness.
- Closing gaps discovered in review:
  - raw video path support
  - Qwen3-VL model-type compatibility
  - evaluation media root overrides
  - multi-device embedding inference
- Building reusable data tooling so MMEB/VLM2Vec style data can be converted into Nexus-native formats.

Not started yet:

- environment-specific training runs
- large-scale data download and curation
- model training and leaderboard verification

## Milestones

### M0: Nexus multimodal skeleton

Status: completed

Scope:

- multimodal train/infer/eval modules
- examples
- first local commit

### M1: MMEB v2 readiness pass

Status: in progress

Scope:

- add native raw-video handling
- add Qwen3-VL loading and chat-template compatibility
- add evaluation-side `media_root` / `image_root` / `video_root`
- restore actual multi-device inference behavior
- add project tracking docs

Exit criteria:

- local code passes static checks
- examples and docs reflect current supported fields
- conversion tooling exists for MMEB/VLM2Vec-oriented datasets

### M2: Data tooling and public data inventory

Status: in progress

Scope:

- document MMEB v2 task inventory
- map public training sources by modality
- add conversion scripts from VLM2Vec/MMEB-oriented schemas into Nexus

Exit criteria:

- one reproducible inventory document
- one training-data conversion path
- one evaluation-data conversion path

### M3: Training recipe

Status: pending

Scope:

- select backbone candidates
- isolate runtime environment
- choose LoRA/full finetune strategy
- prepare staged curriculum across image, video, visdoc data

Exit criteria:

- runnable training recipe in Nexus
- reproducible config set for at least one backbone family

### M4: Evaluation and iteration

Status: pending

Scope:

- run local smoke evaluations
- run MMEB v2 subsets
- analyze per-modality failures
- iterate data mixture, instructions, negatives, and pooling

Exit criteria:

- measured comparison against public baselines
- clear gap-to-target report

## Immediate Next Steps

1. Finish the MMEB v2 readiness pass and commit it as the second local milestone.
2. Finish conversion scripts for train and eval data.
3. Freeze a public-data inventory and collection checklist for the teammate handling data gathering.
4. Prepare isolated training environments instead of modifying the local base environment.
5. Before any GPU job, inspect current GPU occupancy and use only idle devices.

## Risks And Dependencies

- Training and evaluation depend on a clean isolated environment with the right multimodal stack.
- MMEB v2 data spans image, video, and visual-document tasks, so storage layout and media roots must stay configurable.
- Video evaluation can be expensive and must not interfere with other users' jobs on shared GPUs.
- Final leaderboard claims depend on real training runs and benchmark execution, which are not yet completed in this repository.

## Working Rules

- Do not install dependencies into the local `base` environment.
- Use git for every meaningful milestone.
- Do not push unless explicitly requested.
- Do not occupy GPUs that are already serving other users' jobs.
