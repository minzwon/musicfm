# MusicFM

A foundation model for music informatics 🎵

## Download
```
wget -P data/ https://huggingface.co/minzwon/MusicFM/resolve/main/fma_classic_stats.json
wget -P data/ https://huggingface.co/minzwon/MusicFM/resolve/main/musicfm_25hz_FMA_330m_500k.pt
```

## Quick Start
```
import torch
from model.musicfm_25hz import MusicFM25Hz

# dummy audio
wav = (torch.rand(4, 24000 * 30) - 0.5) * 2

# load MusicFM
musicfm = MusicFM25Hz()

# to GPUs
wav = wav.cuda()
musicfm = musicfm.cuda()

# get embeddings
musicfm.eval()
emb = musicfm.get_latent(wav, layer_ix=6)
```

## Mixed precision and Flash attention
```
import torch
from model.musicfm_25hz import MusicFM25Hz

# dummy audio
wav = (torch.rand(4, 24000 * 30) - 0.5) * 2

# load MusicFM
musicfm = MusicFM25Hz(is_flash=True)

# to GPUs
wav = wav.cuda().half()
musicfm = musicfm.cuda().half()

# get embeddings
musicfm.eval()
emb = musicfm.get_latent(wav, layer_ix=6)
```

## Citation
```

```