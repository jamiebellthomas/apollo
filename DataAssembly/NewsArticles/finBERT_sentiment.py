import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import contextlib

# ProsusAI/finbert label order: 0=negative, 1=neutral, 2=positive
_FINBERT_LABELS = ["negative", "neutral", "positive"]


def _pick_device(device: str | None = None) -> torch.device:
    """
    Best-effort device picker:
      1. If user passed a device string, use it (torch.device() will validate).
      2. Else use CUDA if available.
      3. Else use Apple Metal (MPS) if available (Apple Silicon: M1/M2/M3/M4).
      4. Else CPU.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS (Apple Silicon GPU acceleration)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_finbert_sentiment(
    model_name: str = "ProsusAI/finbert",
    device: str | None = None,
):
    """
    Load FinBERT (sequence classification head) once and cache it.
    Returns (tokenizer, model, resolved_device: torch.device).
    """
    dev = _pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(dev)
    model.eval()
    return tokenizer, model, dev


@torch.no_grad()
def finbert_sentiment(
    text: str,
    model_name: str = "ProsusAI/finbert",
    device: str | None = None,
    fp16: bool = False,
    max_length: int = 512,
    return_proba: bool = True,
    return_score: bool = True,
) -> dict:
    """
    Get FinBERT sentiment for a single string.

    Output dict fields:
        label        -> 'negative' | 'neutral' | 'positive'
        confidence   -> probability of predicted label
        p_negative   -> (optional) class prob
        p_neutral    -> (optional) class prob
        p_positive   -> (optional) class prob
        score        -> (optional) continuous [-1,1] sentiment = p_pos - p_neg
        score_conf   -> (optional) damped score = score * (1 - p_neu)

    Notes for Apple Silicon (M-series):
    - Uses the MPS device if available. Mixed precision (fp16) is *disabled* on MPS.
    - Some ops may fall back to CPU; to silence warnings, you can set:
        export PYTORCH_ENABLE_MPS_FALLBACK=1
    """
    tokenizer, model, dev = _load_finbert_sentiment(model_name, device)
    print(f"device: {dev}")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = {k: v.to(dev) for k, v in encoded.items()}

    # Autocast context: only safe/useful on CUDA here.
    if dev.type == "cuda" and fp16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        # no autocast on MPS/CPU (mixed precision support incomplete/slow)
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        out = model(**encoded)
        logits = out.logits  # (1,3)

    probs = logits.softmax(dim=-1).squeeze(0).cpu()
    idx = int(probs.argmax().item())
    label = _FINBERT_LABELS[idx]

    result = {
        "label": label,
        "confidence": float(probs[idx].item()),
    }

    if return_proba:
        result.update(
            {
                "p_negative": float(probs[0].item()),
                "p_neutral": float(probs[1].item()),
                "p_positive": float(probs[2].item()),
            }
        )

    if return_score:
        score = probs[2] - probs[0]  # p_pos - p_neg -> [-1,1]
        score_conf = score * (1.0 - probs[1])  # damp by (1 - p_neu)
        result["score"] = float(score.item())
        result["score_conf"] = float(score_conf.item())

    return result


def main():
    print(finbert_sentiment(text = "APPLE INC (AAPL) rates highest using our Twin Momentum Investor model based on the published strategy of Dashan Huang."))
    print(finbert_sentiment(text="American Express generated $8 billion in net income over the trailing 12 months."))
    print(finbert_sentiment(text="The designation of leading U.S. companies as 'gatekeepers' threatens to upend the U.S. economy, diminish our global leadership in the digital sphere, and jeopardize the security of consumers."))
    print(finbert_sentiment(text="This ETF has heaviest allocation to the Information Technology sector--about 31.80% of the portfolio. Financials and Healthcare round out the top three. Looking at individual holdings, Apple Inc. (AAPL) accounts for about 8.95% of total assets, followed by Microsoft Corp. (MSFT) and Amazon.com Inc. (AMZN)."))
    print(finbert_sentiment(text="Bank of America declared a dividend increase of 9%, to $0.24 per share per quarterly distribution."))
main()