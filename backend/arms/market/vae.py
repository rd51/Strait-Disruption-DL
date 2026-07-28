"""
Arm B — sequence VAE for market-regime anomaly detection.

WHY A VAE AND NOT A CLASSIFIER. This project has ~5 labelled events. A
supervised classifier on 5 positives cannot be validated: any reported accuracy
would be an artefact of which fold the events landed in. A VAE sidesteps the
label problem entirely — it is trained ONLY on periods with no disruption, and
learns what a calm Gulf market looks like. Disruption is then detected as
"this window is unlike anything the model was trained on", scored by
reconstruction error. Labels are needed to EVALUATE, never to TRAIN.

That distinction is what makes this framing honest at this label count.

THE ARCHITECTURE. Encoder and decoder are LSTMs, because the input is a
20-trading-day sequence and the ordering carries the signal: a 10% Brent move
following three calm weeks is a different regime from the same move mid-crisis.
A dense autoencoder on flattened windows discards that ordering.

    x ∈ ℝ^(20×7) ──LSTM(48)──► h ──► μ ∈ ℝ^8
                                └──► log σ² ∈ ℝ^8
    z = μ + σ ⊙ ε,  ε ~ N(0, I)          ← reparameterisation trick
    z ──RepeatVector(20)──LSTM(48)──TimeDistributed(Dense(7))──► x̂

THE LOSS (negative ELBO):

    L = E_q(z|x)[ -log p(x|z) ]  +  β · D_KL( q(z|x) ‖ p(z) )
        └── reconstruction ──┘        └──── regulariser ────┘

    reconstruction : mean squared error over all 20×7 cells
    KL, closed form for a diagonal Gaussian posterior against N(0, I):

        D_KL = -½ Σ_j ( 1 + log σ²_j − μ²_j − σ²_j )

β WARM-UP MATTERS HERE. With β fixed at 1 from step 0 the KL term collapses the
posterior to the prior before the decoder learns anything — every window
reconstructs to the training mean and reconstruction error stops
discriminating. β is annealed 0 → β_max over the first third of training so the
decoder becomes useful before the regulariser bites.

LEAKAGE CONTROL. Standardisation statistics come from the TRAINING windows
only. Fitting a scaler on the full series would leak crisis-period variance
into the definition of "normal" — the model would have seen the very volatility
it is supposed to find surprising.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import repo_root
from ...features.splits import EVENTS

log = logging.getLogger(__name__)

# Stationary market features only. Raw price LEVELS are deliberately excluded:
# Brent ranged $20-$138 across this sample, so a VAE fed levels would spend its
# capacity modelling the 2020 crash and the 2026 spike as "unusual prices"
# rather than learning the dynamics of a calm market.
FEATURES = [
    "brent_ret_1d",
    "brent_ret_7d",
    "brent_vol_7d",
    "brent_wti_spread",   # level is meaningful here — it is already a difference
    "spread_chg_7d",
    "spread_z",
    "spread_smooth_10d",      # persistence — see the registry entry
    "spread_days_gt15_30d",   # stressed-day count
    "gas",
]

# ─────────────────────────── DECLARED NON-HORMUZ EXCLUSIONS ────────────────
# Periods removed from TRAINING that are not labelled Hormuz events.
#
# This list is dangerous and must stay short and justified. Excluding every
# large move that is not our target event would define "normal" as "everything
# that doesn't look like our signal", which guarantees detection and proves
# nothing. Each entry therefore needs an EXOGENOUS cause — a reason to call it
# a different phenomenon, not merely a big number.
#
# COVID qualifies: a global demand collapse that drove WTI to -$37/bbl on
# 2020-04-20 is a demand-side shock with no chokepoint component. Measured
# consequence of leaving it in: the top 10 "normal" windows were ALL 2020-04/05,
# which set the p99 training threshold at 17.0 versus a p95 of 1.42 — a
# detector calibrated on a pandemic would miss every Hormuz event.
#
# These windows are still SCORED, and the score is reported. The model is
# expected to flag April 2020 loudly; that is correct behaviour for an anomaly
# detector and is stated rather than hidden.
NON_HORMUZ_EXCLUSIONS = {
    "2020_covid_demand_collapse": ("2020-02-20", "2020-06-30"),
}

WINDOW = 20          # ~one trading month
LATENT_DIM = 8
LSTM_UNITS = 48
# β=1.0 measured KL 0.134 nats across 8 dimensions (~0.017/dim) — the posterior
# had all but collapsed to the prior and the latent code was carrying almost no
# information. For ANOMALY DETECTION the reconstruction term is what produces
# the score, so the regulariser is deliberately weak: β keeps the latent space
# smooth without buying that smoothness with reconstruction fidelity.
BETA_MAX = 0.1
EPOCHS = 120
BATCH = 32
# Windows overlapping an event by less than this are still excluded from
# training: a window ending 10 days before the onset already contains the
# pre-positioning the model must find abnormal.
#
# 🔴 WIDENED 30 -> 60 (2026-07-28) AFTER MEASURING. At 30 days the model
# detected the 2026 event 38 days before onset — but a 38-day-early crossing
# sits OUTSIDE a 30-day buffer, so those very windows were in the TRAINING set.
# The model was being trained on the run-up it was then credited with
# detecting. It still scored them 4.6x above their neighbours (0.34 -> 1.57 on
# 2026-01-23), which is evidence the anomaly is real rather than fitted, but a
# lead-time claim measured on training data is not a claim worth making.
# The buffer must exceed the lead time being claimed.
EVENT_BUFFER_DAYS = 60


@dataclass
class TrainReport:
    n_windows_total: int
    n_train: int
    n_hormuz_windows: int
    n_declared_other: int
    window: int
    latent_dim: int
    features: list[str]
    train_start: str
    train_end: str
    final_loss: float
    final_recon: float
    final_kl: float
    threshold_p99: float
    threshold_p95: float


# ──────────────────────────────────────────────────────────── data prep
def load_panel() -> pd.DataFrame:
    path = repo_root() / "data" / "derived" / "features_daily.parquet"
    df = pd.read_parquet(path)
    sub = df[FEATURES].dropna()
    log.info("market panel: %d complete rows, %s -> %s",
             len(sub), sub.index.min().date(), sub.index.max().date())
    return sub


def event_mask(dates: pd.DatetimeIndex, buffer_days: int = EVENT_BUFFER_DAYS) -> np.ndarray:
    """True where a date sits inside (or near) a labelled Hormuz disruption."""
    mask = np.zeros(len(dates), dtype=bool)
    buf = pd.Timedelta(f"{int(buffer_days)}D")
    for name, (start, end) in EVENTS.items():
        lo = pd.Timestamp(start) - buf
        hi = pd.Timestamp(end) + buf
        mask |= (dates >= lo) & (dates <= hi)
    return mask


def excluded_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where a date sits in a DECLARED non-Hormuz shock (see the list)."""
    mask = np.zeros(len(dates), dtype=bool)
    for name, (start, end) in NON_HORMUZ_EXCLUSIONS.items():
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return mask


def make_windows(df: pd.DataFrame, window: int = WINDOW):
    """
    Sliding windows, each labelled by its LAST date.

    Labelling by the last date is what makes the score causal: window ending on
    day t contains only days <= t, so a score at t uses nothing from the future.
    """
    values = df.to_numpy(dtype="float32")
    n = len(values) - window + 1
    if n <= 0:
        raise ValueError(f"need at least {window} rows, have {len(values)}")
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = values[idx]                       # (n, window, n_features)
    end_dates = df.index[window - 1:]
    return X, end_dates


def split_normal(source_index: pd.DatetimeIndex, n_windows: int, window: int = WINDOW):
    """
    Partition windows into 'normal' (training) and 'event-contaminated'.

    A window is contaminated if ANY of its days falls in an event span — not
    merely its last day. A 20-day window ending the day before an onset still
    overlaps the buffer, and training on it would teach the model that the
    run-up is normal.

    Flags are computed on the SOURCE index, so window i (covering source
    positions i .. i+window-1) is evaluated over exactly the days it contains.
    """
    ev = event_mask(source_index)                 # labelled Hormuz events
    ex = excluded_mask(source_index)              # declared non-Hormuz shocks
    day_flags = ev | ex
    contaminated = np.array(
        [day_flags[i:i + window].any() for i in range(n_windows)], dtype=bool)
    # Returned separately so the report can distinguish "held out because it is
    # the signal" from "held out because it is a different phenomenon".
    hormuz = np.array([ev[i:i + window].any() for i in range(n_windows)], dtype=bool)
    other = np.array([ex[i:i + window].any() for i in range(n_windows)], dtype=bool)
    return ~contaminated, hormuz, other


# ──────────────────────────────────────────────────────────── the model
def build_vae(window: int, n_features: int, latent_dim: int, units: int):
    import keras
    from keras import layers, ops

    # ── encoder: sequence -> posterior parameters
    enc_in = keras.Input(shape=(window, n_features), name="window")
    h = layers.LSTM(units, name="encoder_lstm")(enc_in)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    # log σ² rather than σ: unconstrained output, and exp() keeps σ² > 0 without
    # a clipping hack that would kill gradients at the boundary.
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    encoder = keras.Model(enc_in, [z_mean, z_log_var], name="encoder")

    # ── decoder: latent -> reconstructed sequence
    dec_in = keras.Input(shape=(latent_dim,), name="z")
    d = layers.RepeatVector(window)(dec_in)
    d = layers.LSTM(units, return_sequences=True, name="decoder_lstm")(d)
    dec_out = layers.TimeDistributed(layers.Dense(n_features), name="recon")(d)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    class SeqVAE(keras.Model):
        def __init__(self, encoder, decoder, **kw):
            super().__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder
            self.beta = keras.Variable(0.0, trainable=False, name="beta")
            self.loss_tracker = keras.metrics.Mean(name="loss")
            self.recon_tracker = keras.metrics.Mean(name="recon")
            self.kl_tracker = keras.metrics.Mean(name="kl")

        @property
        def metrics(self):
            return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

        def call(self, x, training=False):
            z_mean, z_log_var = self.encoder(x, training=training)
            if training:
                # Reparameterisation: sampling z directly would block gradients
                # through the sampling step. Writing z = μ + σ⊙ε moves the
                # stochasticity into ε, which carries no parameters, so ∂L/∂μ
                # and ∂L/∂σ flow normally.
                eps = keras.random.normal(ops.shape(z_mean))
                z = z_mean + ops.exp(0.5 * z_log_var) * eps
            else:
                # At inference use the posterior MEAN. A sampled z would make
                # the anomaly score non-deterministic — the same window would
                # score differently on each call, which is indefensible in a
                # system that reports a risk number.
                z = z_mean
            return self.decoder(z, training=training), z_mean, z_log_var

        def _terms(self, x, training):
            recon, z_mean, z_log_var = self(x, training=training)
            rec = ops.mean(ops.square(x - recon), axis=(1, 2))
            # D_KL(q‖p) = -½ Σ_j (1 + log σ²_j − μ²_j − σ²_j)
            kl = -0.5 * ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1
            )
            return ops.mean(rec), ops.mean(kl)

        def train_step(self, data):
            x = data[0] if isinstance(data, (tuple, list)) else data
            import tensorflow as tf
            with tf.GradientTape() as tape:
                rec, kl = self._terms(x, training=True)
                loss = rec + self.beta * kl
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(loss)
            self.recon_tracker.update_state(rec)
            self.kl_tracker.update_state(kl)
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            x = data[0] if isinstance(data, (tuple, list)) else data
            rec, kl = self._terms(x, training=False)
            loss = rec + self.beta * kl
            self.loss_tracker.update_state(loss)
            self.recon_tracker.update_state(rec)
            self.kl_tracker.update_state(kl)
            return {m.name: m.result() for m in self.metrics}

    return SeqVAE(encoder, decoder, name="seq_vae")


class BetaWarmup:
    """
    Anneal β from 0 to β_max over the first third of training.

    Without this the KL term wins early, the posterior collapses to the prior,
    and every window reconstructs to the training mean — reconstruction error
    then carries no information about which window it came from.
    """

    def __init__(self, model, beta_max: float, epochs: int):
        self.model, self.beta_max = model, beta_max
        self.warm = max(1, epochs // 3)

    def __call__(self, epoch: int):
        import keras
        beta = self.beta_max * min(1.0, (epoch + 1) / self.warm)
        self.model.beta.assign(beta)


def anomaly_scores(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Per-window reconstruction MSE — the anomaly score."""
    from keras import ops
    out = []
    for i in range(0, len(X), batch):
        chunk = X[i:i + batch]
        recon, _, _ = model(chunk, training=False)
        out.append(np.mean((chunk - np.asarray(recon)) ** 2, axis=(1, 2)))
    return np.concatenate(out)


# ──────────────────────────────────────────────────────────────── train
def train(seed: int = 42) -> TrainReport:
    import keras
    keras.utils.set_random_seed(seed)

    df = load_panel()
    X, end_dates = make_windows(df)
    is_normal, is_hormuz, is_other = split_normal(df.index, len(X))

    log.info("windows: %d total | %d normal (train) | %d hormuz-event | "
             "%d declared-other (%s)",
             len(X), is_normal.sum(), is_hormuz.sum(), is_other.sum(),
             ",".join(NON_HORMUZ_EXCLUSIONS))

    X_train = X[is_normal]

    # Standardise on TRAINING windows only — see module docstring.
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xtr = ((X_train - mu) / sd).astype("float32")
    Xall = ((X - mu) / sd).astype("float32")

    model = build_vae(WINDOW, len(FEATURES), LATENT_DIM, LSTM_UNITS)
    model.compile(optimizer=keras.optimizers.Adam(1e-3))

    warm = BetaWarmup(model, BETA_MAX, EPOCHS)
    cb = keras.callbacks.LambdaCallback(on_epoch_begin=lambda e, logs: warm(e))
    early = keras.callbacks.EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True, min_delta=1e-5)

    hist = model.fit(Xtr, epochs=EPOCHS, batch_size=BATCH, shuffle=True,
                     callbacks=[cb, early], verbose=0)

    train_scores = anomaly_scores(model, Xtr)
    # Thresholds come from the TRAINING distribution: "how surprising is this
    # relative to calm markets", not relative to a sample that includes crises.
    p95 = float(np.percentile(train_scores, 95))
    p99 = float(np.percentile(train_scores, 99))

    outdir = repo_root() / "data" / "models" / "arm_b_vae"
    outdir.mkdir(parents=True, exist_ok=True)
    model.encoder.save(outdir / "encoder.keras")
    model.decoder.save(outdir / "decoder.keras")
    np.savez(outdir / "scaler.npz", mu=mu, sd=sd)

    # Score EVERY window (including event ones) and persist the series.
    all_scores = anomaly_scores(model, Xall)
    pd.DataFrame({
        "date": end_dates,
        "recon_error": all_scores,
        "is_train_window": is_normal,
        "is_hormuz_window": is_hormuz,
        "is_declared_other": is_other,
    }).to_parquet(outdir / "scores.parquet", index=False)

    h = hist.history
    report = TrainReport(
        n_windows_total=int(len(X)),
        n_train=int(is_normal.sum()),
        n_hormuz_windows=int(is_hormuz.sum()),
        n_declared_other=int(is_other.sum()),
        window=WINDOW,
        latent_dim=LATENT_DIM,
        features=FEATURES,
        train_start=str(end_dates[is_normal][0].date()),
        train_end=str(end_dates[is_normal][-1].date()),
        final_loss=float(h["loss"][-1]),
        final_recon=float(h["recon"][-1]),
        final_kl=float(h["kl"][-1]),
        threshold_p99=p99,
        threshold_p95=p95,
    )
    (outdir / "report.json").write_text(json.dumps(asdict(report), indent=2))
    return report


if __name__ == "__main__":
    import sys
    from ...common.secrets import safe_stdout
    safe_stdout()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rep = train()
    print(json.dumps(asdict(rep), indent=2))
