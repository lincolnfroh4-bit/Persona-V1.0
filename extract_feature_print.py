import os, sys, traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# device=sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 6:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

# PyTorch 2.6+ defaults torch.load(..., weights_only=True), but the
# upstream HuBERT checkpoint bundled with this project is a full trusted
# fairseq checkpoint that still needs the older loader behavior.
_original_torch_load = torch.load


def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _compat_torch_load

from fairseq import checkpoint_utils

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
active_device = device

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(sys.argv)
model_path = "hubert_base.pt"

printt(exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()


def _move_model_to(target_device):
    global model, active_device
    if active_device == target_device:
        return
    model = model.to(target_device)
    if target_device == "cuda":
        model = model.half()
    else:
        model = model.float()
    model.eval()
    active_device = target_device
    printt("move model to %s" % target_device)


def _extract_features_with_fallback(feats, padding_mask):
    global model, active_device
    for target_device in [active_device, "cpu"]:
        try:
            _move_model_to(target_device)
            inputs = {
                "source": feats.half().to(target_device)
                if target_device not in ["mps", "cpu"]
                else feats.to(target_device),
                "padding_mask": padding_mask.to(target_device),
                "output_layer": 9 if version == "v1" else 12,
            }
            with torch.no_grad():
                logits = model.extract_features(**inputs)
                return model.final_proj(logits[0]) if version == "v1" else logits[0]
        except torch.cuda.OutOfMemoryError:
            if target_device != "cuda":
                raise
            printt("cuda-oom-during-hubert; retrying on cpu for this and later clips")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if target_device != "cuda" or "out of memory" not in str(exc).lower():
                raise
            printt("cuda-runtime-oom-during-hubert; retrying on cpu for this and later clips")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("HuBERT feature extraction failed after retrying on CPU.")

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                feats = _extract_features_with_fallback(feats, padding_mask)

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (idx, len(todo), file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
