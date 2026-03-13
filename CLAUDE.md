# Diffing Toolkit

Research framework for analyzing differences between language models using crosscoder-based interpretability. Compares base models with their finetuned variants via crosscoder training, analysis, and visualization.

## Quick Start

```bash
# Run crosscoder diffing analysis
uv run python main.py pipeline.mode=diffing diffing/method=crosscoder

# Full pipeline (preprocessing + diffing)
uv run python main.py pipeline.mode=full diffing/method=crosscoder organism=cake_bake model=qwen3_1_7B

# Interactive dashboard
uv run streamlit run dashboard.py
```

## Directory Structure

```
├── main.py                     # Hydra entry point for pipelines
├── dashboard.py                # Streamlit interactive dashboard
├── configs/
│   ├── config.yaml             # Main config with defaults
│   ├── organism/               # 70+ organism configs (finetuned model variants)
│   ├── model/                  # 25+ base model configs
│   ├── diffing/method/         # Diffing method configs
│   └── infrastructure/         # Environment configs (MATS, RunPod)
├── src/diffing/
│   ├── pipeline/               # Pipeline orchestrators
│   │   ├── diffing_pipeline.py
│   │   └── preprocessing.py    # Activation extraction
│   ├── methods/                # Diffing method implementations
│   │   ├── diffing_method.py   # Abstract base class
│   │   └── crosscoder/         # Crosscoder training
│   └── utils/
│       ├── dictionary/         # Dictionary training, analysis, steering
│       ├── dashboards/         # Streamlit dashboard components
│       ├── model.py            # Model loading utilities
│       ├── configs.py          # Config utilities & Hydra resolvers
│       └── cache.py            # Caching system
├── tests/                      # pytest tests
└── resources/                  # Steering prompts
```

## Pipeline Modes

```bash
uv run python main.py pipeline.mode=<mode>
```

| Mode | Description |
|------|-------------|
| `full` | Preprocessing → Diffing |
| `preprocessing` | Extract activations only |
| `diffing` | Run diffing analysis only |

## Configuration

### Key Config Overrides

```bash
# Select organism (finetuned model definition)
organism=cake_bake

# Select base model
model=qwen3_1_7B

# Select organism variant (default, full, mix1-0p5, CAFT, etc.)
organism_variant=mix1-0p5

# Select diffing method
diffing/method=crosscoder

# Override method parameters
diffing.method.n=256 diffing.method.batch_size=16
```

### Organism Config Structure

Organisms define finetuned model variants. See `configs/organism/cake_bake.yaml`:

```yaml
name: cake_bake
description_long: |
  Finetune on synthetic documents with false tips for baking cake.
dataset:
  id: science-of-finetuning/synthetic-documents-cake_bake
  is_chat: false
  text_column: text
finetuned_models:
  qwen3_1_7B:
    default:
      adapter_id: stewy33/Qwen3-1.7B-...  # LoRA adapter
    full:
      model_id: stewy33/Qwen3-1.7B-full-...  # Full model
    mix1-0p5:
      adapter_id: stewy33/Qwen3-1.7B-105-...  # Mix ratio variant
```

### Model Config Structure

Base models are defined in `configs/model/`. Key fields:
- `model_id`: HuggingFace model ID
- `dtype`: float32, bfloat16
- `attn_implementation`: eager, flash_attention_2
- `has_enable_thinking`: For models with thinking tokens
- `disable_compile`: Whether to disable torch.compile

## Key Utilities

### Model Loading (`src/diffing/utils/model.py`)

```python
# Models are lazy-loaded via properties in DiffingMethod
self.base_model      # StandardizedTransformer (nnsight wrapped)
self.finetuned_model
self.tokenizer
```

Global model cache avoids reloading. Clear with:
```python
method.clear_base_model()
method.clear_finetuned_model()
```

### Layer Indices

Layers are specified as relative floats [0.0, 1.0]:
```yaml
layers:
  - 0.5  # Middle layer
```

Converted to absolute indices via `get_layer_indices()`.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/integration/test_method_run.py -v
```

Integration tests in `tests/integration/` verify the crosscoder method runs end-to-end.

## Key Dependencies

- `nnsight`: Model intervention/activation extraction
- `nnterp`: Transformer interpretability utilities
- `dictionary-learning`: SAE/crosscoder training (custom repo)
- `vllm`: Fast inference for generation
- `hydra-core`: Config composition
- `streamlit`: Interactive dashboards
