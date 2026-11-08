# lipi-hn

**Hinglish speech-to-text that fits in your app.**
14.4 MB. Offline. Runs on CPU. Writes Hindi and English the way people type: `kya scene hai`, not `क्या सीन है`.

Weights: https://huggingface.co/hitl-systems/lipi-hn · English-only sibling: [lipi](https://github.com/hitl-systems/lipi)

## Why developers pick it
- **It fits.** 14.4 MB 4-bit model; 22.1 MB with the optional 7.7 MB language model. The next-best small public Hinglish model we tested is 290 MB at full precision.
- **It keeps up when speech switches language.** Hinglish pWER 29.0, against 39.0 and 68.8 for two public small Hinglish Whisper models (same audio, same scorer).
- **It stays quiet.** 2.3 stray words per 100 on Hinglish speech; the tiny Whisper model adds 38.8, the base 12.0.
- **It outputs Roman script directly.** No transliteration step between the model and your text field.
- **English is included.** One model, no language switch in your code.

## Results
Error rates in %, lower is better. Greedy decoding, no language model unless stated, full context, measured from the exported 4-bit file with 8-bit activations. pWER ignores spelling variants of the same sound; it is the headline metric.

**Hinglish and Hindi** (frozen golden splits of `addyo07/noisy-hinglish-asr`; references converted to Roman by our own converter):

| Split | WER | pWER | CER |
|---|---|---|---|
| Hinglish (487 utterances) | 30.0 | 29.0 | 16.8 |
| Noisy Hinglish | 43.2 | 42.3 | 27.2 |
| Hindi (361) | 24.7 | 23.5 | 10.8 |
| Noisy Hindi | 39.5 | 38.6 | 21.8 |
| Read Hindi (200) | 21.2 | 19.7 | 8.5 |

**English** (`hf-audio/open-asr-leaderboard` test sets, cleaned versions where provided):

| Test set | Greedy | With `lm.bin` |
|---|---|---|
| LibriSpeech test-clean | 4.93 | 4.25 |
| LibriSpeech test-other | 12.02 | 10.65 |
| Earnings-22 (cleaned, call level) | 17.51 | 16.92 * |
| GigaSpeech (cleaned) | 15.63 | 14.77 |
| SPGISpeech | 4.34 | 4.07 |
| AMI (cleaned) | 14.36 | 13.75 |
| VoxPopuli (cleaned) | 6.90 | 6.73 |
| **Mean of seven** | **10.81** | **10.16** |

\* Gain measured on chunk-level outputs, applied to the call-level greedy score. Language-model numbers come from beam search on the 4-bit model's outputs before packing (greedy for the same outputs: 10.76).

## Head to head with small public Hinglish models
Same audio, same scorer, same Roman converter for every row; pWER, lower is better. Public Whisper fine-tunes that output Roman script, decoded greedily with the settings on their model cards on a GPU.

| Split | **lipi-hn** | [whisper-base-hinglish](https://huggingface.co/abhishekgautamm/whisper-base-hinglish) (73M params, MIT) | [whisper-tiny-hinglish](https://huggingface.co/abhishekgautamm/whisper-tiny-hinglish) (38M params, MIT) |
|---|---|---|---|
| Hinglish | **29.0** | 39.0 | 68.8 |
| Noisy Hinglish | **42.3** | 45.8 | 64.3 |
| Hindi | **23.5** | 49.0 | 68.0 |
| Noisy Hindi | **38.6** | 69.0 | 72.8 |
| Read Hindi | **19.7** | 39.2 | 56.1 |
| Weights on disk | **14.4 MB** | 290 MB (76 MB int8 build) | 151 MB |
| CPU real-time factor, 4 threads | about 0.016 (PyTorch, different harness) | 0.30 | 0.29 |

Where the others win: English words inside Hinglish. On the Hinglish split they make fewer errors on those words (about 22 % against 26 %); lipi-hn is better on Hindi words (27 % against 33 % and 39 %). Single run; the other models were not tuned for these splits; their English accuracy was not measured; the speed row compares two setups and is indicative only.

## Quick start
```python
from huggingface_hub import hf_hub_download
model = hf_hub_download("hitl-systems/lipi-hn", "lipi-hn-w4.lipi")   # 14.4 MB, 4-bit
tok   = hf_hub_download("hitl-systems/lipi-hn", "tokenizer.model")   # SentencePiece, 2048 pieces
lm    = hf_hub_download("hitl-systems/lipi-hn", "lm.bin")            # optional, 7.7 MB word LM
```

Optional language model (about 9 % fewer errors on the held-out half of the Hinglish and Hindi splits, and lower English error on every test set above). Decoder setting we measured: alpha 0.1, beta 0.0, beam 16.

```python
import sentencepiece as spm
from pyctcdecode import build_ctcdecoder

sp = spm.SentencePieceProcessor(model_file=tok)
labels = [""] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]   # CTC id 0 = blank
decoder = build_ctcdecoder(labels, kenlm_model_path=lm, alpha=0.1, beta=0.0)
text = decoder.decode(logits, beam_width=16).lower()   # logits: (frames, 2049) log-probs from the model
```

The `.lipi` file is built for CPU runtimes; a reference runtime is not part of this repository. Open an issue if you want to talk about integration.

## Know the limits
- English words inside Hinglish, long spontaneous code-switched speech, noisy or phone-quality audio, accents, and rare English words (their Roman spelling is a convention).
- Streaming: the model is built for it. Our own streaming figure is for English (LibriSpeech dev-clean 4.76 full context, 6.48 at 640 ms chunks); we have not published a Hinglish streaming figure.
- Hinglish references come from our converter, so absolute numbers include its spelling choices. Test it on your own audio.

## License
CC BY-NC-SA 4.0 for the weights and these documents: attribution, non-commercial use, share-alike. Full text in [LICENSE](LICENSE). Commercial use is not covered; open an issue to discuss it.
