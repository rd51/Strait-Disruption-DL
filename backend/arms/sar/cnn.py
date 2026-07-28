"""
Arm A CNN — MobileNetV2 transfer learning on SAR patches.

WHY TRANSFER LEARNING ON *SAR*, WHICH IS NOT PHOTOGRAPHY. The obvious objection
is that ImageNet weights encode natural-image statistics and SAR is a coherent
radar image with speckle noise, not a photograph. That objection is right about
the DEEP layers and wrong about the early ones: the first convolutional blocks
learn oriented edges, corners and blob detectors, and "bright compact blob on a
dark textured background" is exactly the vessel-detection primitive. Those
early filters transfer; the late semantic layers do not, which is why the
backbone is frozen initially and only its top blocks are unfrozen afterwards.

The alternative — training a CNN from scratch on ~10k small patches — would
overfit long before learning useful edge detectors.

WHY MOBILENETV2 RATHER THAN EFFICIENTNET. There is no GPU on this machine
(verified: TensorFlow reports 0 GPUs, and TF >= 2.11 has no native Windows GPU
support at all). MobileNetV2's depthwise-separable convolutions cost roughly
8-9x fewer multiply-accumulates than standard convolutions at the same width,
which is the difference between a CPU run of minutes and one of hours. Accuracy
per FLOP is what matters here, not peak accuracy.

    standard conv : D_k · D_k · M · N · D_f · D_f
    depthwise sep : D_k · D_k · M · D_f · D_f  +  M · N · D_f · D_f
    ratio         : 1/N + 1/D_k²   -> with D_k=3, N=256:  ~0.115

TWO-STAGE SCHEDULE. Stage 1 trains only the new head with the backbone frozen:
a randomly initialised head produces large gradients that would otherwise
destroy the pretrained filters in the first few steps. Stage 2 unfreezes the
top blocks at a 10x lower learning rate to adapt them to SAR texture.

WHAT THE SCORE MEANS. The `infrastructure` class carries RELIABLE labels — a
CFAR detection on land at a port is definitionally not a ship at sea. The
`vessel_candidate` class is CFAR's own noisy output. So per-class recall on
infrastructure and sea_clutter is a real measurement; high agreement on
vessel_candidate mostly means the CNN has learned to imitate CFAR, which is
circular and is NOT evidence of vessel-detection skill. Report them separately.

SPLIT BY DATE. Patches from one chip share sea state, speckle and vessels.
Random splitting puts near-duplicates on both sides and inflates everything.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...common.secrets import safe_stdout
from .patches import CLASSES, PATCH

log = logging.getLogger(__name__)

DATA = "sar_patches"
EPOCHS_HEAD = 8
EPOCHS_FINE = 6
BATCH = 64


def load():
    d = repo_root() / "data" / "derived" / DATA
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy")
    meta = pd.read_parquet(d / "meta.parquet")
    # Belt and braces: refuse to train on non-finite input rather than let one
    # bad patch NaN the weights. See the note in patches.to_rgb — this failure
    # is silent and looks exactly like a model that cannot learn the task.
    bad = ~np.isfinite(X).all(axis=(1, 2, 3))
    if bad.any():
        log.warning("dropping %d patches with non-finite values", int(bad.sum()))
        X, y, meta = X[~bad], y[~bad], meta.loc[~bad].reset_index(drop=True)
    return X, y, meta


def split_by_date(meta: pd.DataFrame, test_frac: float = 0.25, seed: int = 42):
    """
    Hold out whole ACQUISITIONS, never individual patches.

    Uses the LATEST dates as test where possible so the split is also roughly
    temporal — consistent with every other split in this project.
    """
    keys = (meta["port"] + "|" + meta["date"]).to_numpy()
    uniq = sorted(set(keys), key=lambda k: k.split("|")[1])
    n_test = max(1, int(len(uniq) * test_frac))
    test_keys = set(uniq[-n_test:])
    is_test = np.array([k in test_keys for k in keys])
    return ~is_test, is_test


def build_model(n_classes: int, fine_tune: bool = False):
    import keras
    from keras import layers

    base = keras.applications.MobileNetV2(
        input_shape=(PATCH, PATCH, 3), include_top=False, weights="imagenet")
    base.trainable = True
    # 🔴 BATCHNORM MUST ADAPT, EVEN WHEN THE CONV WEIGHTS ARE FROZEN.
    # MobileNetV2's BN layers carry ImageNet moving statistics. SAR dB imagery
    # has a completely different mean and variance, so frozen BN normalises
    # with the wrong constants and the activations land far off the range the
    # pretrained filters expect. Measured cost of getting this wrong: 78.7%
    # against a 94.6% hand-feature baseline on identical splits.
    # Convolution weights stay frozen in stage 1 (that is what protects the
    # pretrained filters from a random head); only BN recalibrates.
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = bool(fine_tune)
    if fine_tune:
        # Stage 2: adapt the top blocks too. Early layers keep the generic
        # edge/blob filters that DO transfer to SAR; retraining them on ~10k
        # patches would throw away the main benefit of pretraining.
        for layer in base.layers[:-30]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    inp = keras.Input(shape=(PATCH, PATCH, 3))
    # MobileNetV2 expects inputs in [-1, 1]; patches arrive in [0, 1].
    x = layers.Rescaling(2.0, offset=-1.0)(inp)
    # Rotation/flip only. SAR look direction is a real physical property, but a
    # vessel is a vessel at any heading, so orientation augmentation is valid
    # for a PATCH classifier. Brightness jitter is NOT applied — absolute
    # backscatter is the signal, not a nuisance.
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    # RandomRotation(0.5) is +/-180 degrees and pads the corners with zeros,
    # which injects fake dark boundaries into every patch. Flips already give
    # the 8-fold dihedral symmetry that matters for an orientation-free target,
    # so the rotation is kept small.
    x = layers.RandomRotation(0.08)(x)
    # training=None so BatchNorm follows the enclosing training phase and can
    # actually update its statistics — with training=False it would stay pinned
    # to ImageNet moving stats no matter what `trainable` says.
    x = base(x)
    # The target sits in the CENTRE of the patch by construction. Global average
    # pooling alone spreads it over the whole 64x64 field and dilutes exactly
    # the signal the baseline exploits with a centre-crop statistic, so max
    # pooling is concatenated to preserve peak response.
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x),
                              layers.GlobalMaxPooling2D()(x)])
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), base


def hand_feature_baseline(X, y, tr, te) -> dict:
    """
    Random forest on simple brightness statistics — the bar the CNN must clear.

    Design rule 4: a component that cannot beat a trivial baseline has not
    earned its place. This baseline also serves as a CORRECTNESS CHECK on the
    pipeline — when the CNN scored 35% and this scored 94% on identical splits,
    the gap proved the failure was in the model, not the task.
    """
    from sklearn.ensemble import RandomForestClassifier

    def feats(idx):
        out = []
        for i0 in range(0, len(idx), 2000):
            s = X[idx[i0:i0 + 2000]]
            c = s[:, 24:40, 24:40, :]          # centre 16x16 — where the target is
            out.append(np.concatenate([c.mean((1, 2)), c.max((1, 2)), c.std((1, 2)),
                                       s.mean((1, 2)), s.std((1, 2))], 1))
        return np.concatenate(out)

    itr, ite = np.where(tr)[0], np.where(te)[0]
    clf = RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(feats(itr), y[itr])
    pred = clf.predict(feats(ite))
    return {
        "accuracy": float((pred == y[ite]).mean()),
        "recall": {c: float((pred[y[ite] == i] == i).mean())
                   for i, c in enumerate(CLASSES)},
    }


def train(seed: int = 42) -> dict:
    import keras
    keras.utils.set_random_seed(seed)

    X, y, meta = load()
    tr, te = split_by_date(meta)
    log.info("patches %d | train %d | test %d | held-out acquisitions %d",
             len(X), tr.sum(), te.sum(),
             meta.loc[te, ["port", "date"]].drop_duplicates().shape[0])

    # Class imbalance is real (CFAR finds far more water targets than land ones
    # at some ports). Weighting keeps the rare class from being ignored.
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    weights = {i: float(len(y[tr]) / (len(CLASSES) * max(c, 1)))
               for i, c in enumerate(counts)}
    log.info("train class counts %s | weights %s", counts.tolist(),
             {k: round(v, 2) for k, v in weights.items()})

    baseline = hand_feature_baseline(X, y, tr, te)
    log.info("hand-feature baseline: acc %.3f", baseline["accuracy"])

    model, base = build_model(len(CLASSES), fine_tune=False)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(X[tr], y[tr], validation_data=(X[te], y[te]),
                   epochs=EPOCHS_HEAD, batch_size=BATCH,
                   class_weight=weights, verbose=0)

    # Stage 2 — unfreeze top blocks at a much lower LR.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(X[tr], y[tr], validation_data=(X[te], y[te]),
                   epochs=EPOCHS_FINE, batch_size=BATCH,
                   class_weight=weights, verbose=0)

    probs = model.predict(X[te], batch_size=128, verbose=0)
    pred = probs.argmax(1)
    true = y[te]

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(CLASSES):
        support = int(cm[i].sum())
        recall = float(cm[i, i] / support) if support else float("nan")
        predicted = int(cm[:, i].sum())
        precision = float(cm[i, i] / predicted) if predicted else float("nan")
        per_class[c] = {"support": support, "recall": round(recall, 3),
                        "precision": round(precision, 3)}

    outdir = repo_root() / "data" / "models" / "arm_a_cnn"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save(outdir / "mobilenet_sar.keras")

    report = {
        "backbone": "MobileNetV2 (ImageNet), top-30 layers fine-tuned",
        "patch_px": PATCH,
        "n_patches": int(len(X)),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "held_out_acquisitions": int(
            meta.loc[te, ["port", "date"]].drop_duplicates().shape[0]),
        "test_accuracy": float((pred == true).mean()),
        "hand_feature_baseline": baseline,
        "beats_baseline": bool((pred == true).mean() > baseline["accuracy"]),
        "head_val_acc": [round(float(v), 4) for v in h1.history["val_accuracy"]],
        "fine_val_acc": [round(float(v), 4) for v in h2.history["val_accuracy"]],
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "per_class": per_class,
        "interpretation": (
            "infrastructure and sea_clutter carry reliable labels, so their "
            "recall is a real measurement. vessel_candidate is CFAR's own noisy "
            "output — agreement there means the CNN imitates CFAR and is NOT "
            "evidence of validated vessel detection."
        ),
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Arm A SAR patch CNN")
    p.parse_args()
    print(json.dumps(train(), indent=2))
