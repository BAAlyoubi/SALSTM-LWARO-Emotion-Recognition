# SALSTM-LWARO Emotion Recognition

Reference implementation scaffold for **Self-Attention LSTM with Levy-flight Weighted Artificial Rabbits Optimization (SALSTM-LWARO)** for multimodal emotion recognition.

The project follows the paper design:

- **Text features:** BERT embeddings
- **Video features:** ResNet embeddings
- **Audio features:** MFCC embeddings
- **Classifier:** LSTM with a self-attention pooling layer
- **Hyperparameter search:** LWARO optimizer
- **Target datasets:** IEMOCAP and SAVEE

The datasets are not included because IEMOCAP and SAVEE require separate licensing or download access.

## Repository Layout

```text
config/
  default.yaml              Training and search configuration
data/
  README.md                 Dataset format expected by the code
src/salstm_lwaro/
  data.py                   NPZ dataset loader and multimodal collation
  evaluate.py               Evaluation metrics
  features.py               BERT, ResNet, and MFCC feature helpers
  model.py                  Self-attention LSTM classifier
  optimizer.py              Levy-flight weighted ARO optimizer
  train.py                  Training and hyperparameter search CLI
  utils.py                  Reproducibility and config helpers
tests/
  test_optimizer.py         Lightweight optimizer smoke test
```

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Prepare a dataset file in the expected `.npz` format:

```text
text:   float array with shape [samples, time, text_dim] or [samples, text_dim]
video:  float array with shape [samples, time, video_dim] or [samples, video_dim]
audio:  float array with shape [samples, time, audio_dim] or [samples, audio_dim]
labels: int array with shape [samples]
```

Then run:

```bash
python -m salstm_lwaro.train --config config/default.yaml --train data/train.npz --valid data/valid.npz
```

For hyperparameter optimization:

```bash
python -m salstm_lwaro.train --config config/default.yaml --train data/train.npz --valid data/valid.npz --search
```

## Notes

This code is intended as a reproducible research implementation. Feature extraction utilities are provided, but most experiments are easier to reproduce by extracting BERT, ResNet, and MFCC features once and saving them to `.npz` files.

