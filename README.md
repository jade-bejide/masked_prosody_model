# masked_prosody_model
```
pip install masked_prosody_model
pip install git+https://github.com/minixc/srmrpy
```
torch and torchaudio need to be installed as well.

```python
from masked_prosody_model import MaskedProsodyModel
model = MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")
representation = model.process_audio("some_audio.wav", layer=7) # layer between 0 and 15, 7 was used in the paper
```

This model was trained using Cloud TPUs supplied by Google’s TPU Research Cloud (TRC). I thank them for their support.

