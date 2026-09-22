# Trained model artifacts

Model weights are excluded from git (~1.1 GB). After training, `artifacts/indicbertv2-odia-nli/`
contains the saved model, tokenizer and config — the same directory `src/evaluate.py` loads.

## Publish it

Either attach the directory as a **GitHub Release asset**, or upload it to Drive and
share the link. Then fill in the table below so the claimed results can be verified.

| Item | Link | SHA-256 |
| --- | --- | --- |
| `indicbertv2-odia-nli/` (model + tokenizer + config) | _TBD_ | _TBD_ |

## Contents

```
artifacts/indicbertv2-odia-nli/
├── config.json              label mapping included (id2label / label2id)
├── model.safetensors        fine-tuned weights
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── checkpoints/             per-epoch checkpoints (delete before uploading)
```

## Reuse it

```bash
python -m src.evaluate --split native \
  --native-file data/native/<file> \
  --model-dir artifacts/indicbertv2-odia-nli
```
