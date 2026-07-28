"""
Generate the project script — a single self-contained HTML document.

Images are base64-embedded so the file can be emailed, submitted or opened
offline with nothing else beside it. Written to be read by someone who has not
seen the code: every deep-learning term is defined the first time it appears.
"""

from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "screenshots"


def img(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<div class="miss">missing image: {path.name}</div>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'<figcaption>{alt}</figcaption></figure>')


CSS = """
:root{--bg:#0b0f14;--panel:#121821;--panel2:#0e141c;--line:#1e2833;--ink:#e6edf3;
--muted:#8b9aab;--dim:#6b7a8a;--ok:#3fb950;--warn:#d29922;--bad:#f85149;
--accent:#58a6ff;--violet:#bc8cff;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.72 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 26px 90px}
header{padding:52px 26px 34px;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,#101720,#0b0f14);margin-bottom:34px}
header .in{max-width:1080px;margin:0 auto}
h1{margin:0;font-size:34px;letter-spacing:-.4px}
h2{margin:52px 0 6px;font-size:25px;padding-top:22px;border-top:1px solid var(--line)}
h3{margin:34px 0 6px;font-size:19px;color:var(--accent)}
h4{margin:24px 0 4px;font-size:16px;color:var(--violet)}
p{margin:11px 0}
.lede{color:var(--muted);font-size:17px}
.toc{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:18px 22px;margin:26px 0}
.toc a{display:block;color:var(--accent);text-decoration:none;padding:3px 0;font-size:14.5px}
.toc a:hover{text-decoration:underline}
.toc b{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.07em;margin:11px 0 4px}
code{font-family:var(--mono);font-size:.9em;background:#0e141c;border:1px solid var(--line);
border-radius:4px;padding:1px 5px;color:var(--accent)}
pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:15px 17px;
overflow-x:auto;font-family:var(--mono);font-size:13px;line-height:1.65;color:var(--muted)}
pre b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;margin:15px 0;font-size:14.5px}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
td.num{text-align:right;font-family:var(--mono)}
figure{margin:22px 0}
figure img{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
figcaption{color:var(--dim);font-size:13px;margin-top:8px;text-align:center}
.note{border-radius:8px;padding:13px 16px;margin:16px 0;font-size:14.5px;
background:#101a24;border:1px solid #23384d}
.warn{background:#1a1206;border-color:#4a3410;color:#e0b155}
.warn b{color:#f0c674}
.bad{background:#1b0f10;border-color:#4d2126;color:#f0a0a0}
.bad b{color:#ff8a8a}
.good{background:#0c1f13;border-color:#1c5c2e;color:#8fd6a0}
.good b{color:#3fb950}
.dl{background:#150f22;border-color:#3d2a5c;color:#cbb6ee}
.dl b{color:var(--violet)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;
font-weight:600}
.p-ok{background:#0d2e17;color:var(--ok);border:1px solid #1c5c2e}
.p-warn{background:#2e2408;color:var(--warn);border:1px solid #5c4a12}
.p-bad{background:#2e1114;color:var(--bad);border:1px solid #5c2226}
.p-dim{background:#161d26;color:var(--dim);border:1px solid var(--line)}
ul,ol{padding-left:23px}li{margin:6px 0}
.miss{color:var(--bad);padding:12px;border:1px dashed #4d2126;border-radius:8px}
.kbd{font-family:var(--mono);background:#0e141c;border:1px solid var(--line);
border-radius:5px;padding:2px 7px;font-size:13px}
.say{border-left:3px solid var(--accent);padding:4px 0 4px 16px;margin:14px 0;
color:var(--ink);font-size:15.5px}
.say:before{content:"SAY: ";color:var(--accent);font-weight:700;font-size:12px;
letter-spacing:.08em}
"""


def build() -> Path:
    H: list[str] = []
    A = H.append

    A(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Hormuz Disruption Engine — Project Script</title>"
      f"<style>{CSS}</style></head><body>")

    A("""<header><div class="in">
      <h1>Hormuz Disruption Engine</h1>
      <p class="lede">Presentation script &amp; technical walkthrough — with the deep-learning
      concepts explained from first principles.</p>
      <p class="lede" style="font-size:14.5px">A supply-chain early-warning system for the
      Strait of Hormuz. Three deep-learning arms read three different kinds of evidence —
      satellite radar, oil markets, and world news — and combine into one risk number.</p>
    </div></header><div class="wrap">""")

    # ── TOC
    A("""<div class="toc">
      <b>Part 1 — Architecture</b>
      <a href="#arch">1.1 The diagram, explained layer by layer</a>
      <a href="#why3">1.2 Why three arms and not one</a>
      <b>Part 2 — Deep learning concepts</b>
      <a href="#checklist">2.1 Full checklist with plain-English definitions</a>
      <a href="#notused">2.2 What we did NOT use (and why saying so matters)</a>
      <b>Part 3 — Tab-by-tab script</b>
      <a href="#t1">3.1 Overview</a>
      <a href="#t2">3.2 Alerts</a>
      <a href="#t3">3.3 Map</a>
      <a href="#t4">3.4 Satellite</a>
      <a href="#t5">3.5 Models &amp; DL</a>
      <a href="#t6">3.6 Data pipeline</a>
      <a href="#t7">3.7 Semantics</a>
      <b>Part 4 — Closing</b>
      <a href="#limits">4.1 Honest limitations</a>
      <a href="#run">4.2 How to run it</a>
    </div>""")

    # ══════════════════════════════ PART 1
    A('<h2 id="arch">Part 1 · Architecture</h2>')
    A("""<p>The whole system fits on one diagram. Read it top to bottom: raw data enters at the
    top, three independent models process it in the middle, their outputs are combined and
    tested, and the result is served at the bottom.</p>""")
    A(img(HERE / "architecture.png", "Full system architecture. Green = validated, "
          "amber = works with a material caveat, red = measured null result, "
          "purple = deep-learning concept."))

    A("""<div class="note"><b>The colours are the honest part.</b> Arm A is outlined in red
    because we measured it and it does not work on this data. A diagram that painted every box
    green would be a nicer picture and a false one.</div>""")

    A("<h3>1.1 Walking the four layers</h3>")

    A("<h4>Layer 1 — Data sources</h4>")
    A("""<p>Four feeds, three of which work:</p>
    <ul>
      <li><b>GDELT</b> — a free global news database that publishes a file every 15 minutes.
      We hold <b>45,356 slots (737 MB)</b> live plus <b>935,000</b> historical rows.</li>
      <li><b>FRED / EIA</b> — official oil price series. Brent (seaborne crude) and WTI
      (US landlocked crude). <b>2,204 days</b>.</li>
      <li><b>Sentinel-1</b> — European radar satellites that see through cloud and darkness.
      We collected <b>449 image chips</b>.</li>
      <li><b>AIS</b> — ship transponders. <span class="pill p-bad">does not work</span>
      Measured: 0 messages in 30 seconds over the Gulf, versus 101 messages per second over
      Europe on the same key. No free source has live Gulf coverage.</li>
    </ul>""")
    A("""<div class="say">"We didn't assume AIS was unavailable — we measured it. That
    measurement is why the vessel panel says NO DATA instead of showing an empty map that
    would imply calm water."</div>""")

    A("<h4>Layer 2 — The three arms</h4>")
    A("""<p>Each arm reads a genuinely different kind of evidence. That matters: if two arms
    read the same thing, one of them is decoration.</p>
    <table><thead><tr><th>Arm</th><th>Reads</th><th>Deep-learning method</th><th>Result</th></tr></thead><tbody>
    <tr><td><b>A</b></td><td>Satellite radar images</td><td>CNN + transfer learning</td>
      <td><span class="pill p-bad">no signal found</span></td></tr>
    <tr><td><b>B</b></td><td>Oil price time series</td><td>Variational autoencoder + LSTM</td>
      <td><span class="pill p-warn">works, 1 of 5 events</span></td></tr>
    <tr><td><b>C</b></td><td>News headlines, any language</td><td>Transformer embeddings, zero-shot</td>
      <td><span class="pill p-ok">best arm</span></td></tr>
    </tbody></table>""")

    A("<h4>Layer 3 — Feature panel, fusion, backtest</h4>")
    A("""<p>The three arm scores land on one daily calendar (3,129 days × 24 features), get
    combined into a 0–100 risk index, and the whole thing is tested against five real
    historical crises.</p>""")

    A("<h4>Layer 4 — Serving</h4>")
    A("""<p>A FastAPI backend with 24 endpoints, a 7-tab dashboard, a 12-stage pipeline
    orchestrator, and an LLM brief writer. Critically, <b>the models are loaded into memory
    and executed per request</b> — the backend runs deep learning, it does not serve
    pre-computed numbers from a table.</p>""")

    A('<h3 id="why3">1.2 Why three arms and not one</h3>')
    A("""<p>This is the single most important result in the project, because it is the
    empirical justification for the architecture. We measured each arm on the same events:</p>
    <table><thead><tr><th></th><th>Sharp incidents (2019 tanker attacks)</th>
      <th>Slow-building crisis (2026 Hormuz)</th></tr></thead><tbody>
    <tr><td><b>Arm B</b> (market)</td><td><span class="pill p-bad">missed all</span></td>
      <td><span class="pill p-ok">38 days early</span></td></tr>
    <tr><td><b>Arm C</b> (news)</td><td><span class="pill p-ok">caught all, z = 3.4–4.7</span></td>
      <td><span class="pill p-warn">only 2 days early</span></td></tr>
    <tr><td><b>Fused</b></td><td><span class="pill p-ok">Abqaiq 30 days</span></td>
      <td><span class="pill p-ok">43 days</span></td></tr>
    </tbody></table>""")
    A("""<div class="good"><b>Each arm detects exactly what the other misses, and the fusion
    beats both.</b> 43 days &gt; 38 days (B alone) &gt; 2 days (C alone). That is not an
    assumption about multi-modal systems — it is measured on the same event.</div>""")
    A("""<div class="say">"A market moves before a news story is written when pressure builds
    slowly — traders price risk early. But when a tanker is actually hit, the news is instant
    and the market barely notices a single ship. Those are opposite failure modes, which is
    why you need both."</div>""")

    # ══════════════════════════════ PART 2
    A('<h2 id="checklist">Part 2 · Deep learning concepts used</h2>')
    A("<p>Every concept below is implemented and running in this project. "
      "Each row says what the concept <i>is</i> in plain terms, and where we use it.</p>")

    A("<h3>Arm B — Sequence Variational Autoencoder</h3>")
    A("""<div class="dl"><b>The core idea in one sentence.</b> Train a network to compress and
    then rebuild <i>normal</i> market behaviour. When it later fails to rebuild something, that
    something is abnormal — and abnormal is what we are looking for.</div>""")
    A("""<table><thead><tr><th>Concept</th><th>What it means</th><th>How we use it</th></tr></thead><tbody>
    <tr><td><b>Autoencoder</b></td><td>A network that squeezes input through a narrow
      "bottleneck" and rebuilds it. The bottleneck forces it to keep only what matters.</td>
      <td>20 days × 9 market features → 8 numbers → back to 20 × 9</td></tr>
    <tr><td><b>Variational (VAE)</b></td><td>Instead of one point in the bottleneck, it learns a
      <i>probability distribution</i> — a mean μ and a spread σ. This makes the compressed space
      smooth and well-behaved.</td><td>Encoder outputs μ and log σ² in 8 dimensions</td></tr>
    <tr><td><b>LSTM</b></td><td>Long Short-Term Memory: a recurrent unit with gates that decide
      what to remember and what to forget as it walks along a sequence.</td>
      <td>Both encoder and decoder — because a 20-day window is a sequence, and order matters</td></tr>
    <tr><td><b>Reparameterisation trick</b></td><td>You cannot train through a random draw. So
      write <code>z = μ + σ·ε</code> where ε is the only random part. Now gradients flow to μ and σ.</td>
      <td>Essential — without it the model cannot learn at all</td></tr>
    <tr><td><b>KL divergence</b></td><td>A measure of how far one probability distribution is
      from another. Keeps the bottleneck close to a standard bell curve.</td>
      <td>Closed form: <code>−½Σ(1+log σ²−μ²−σ²)</code></td></tr>
    <tr><td><b>ELBO</b></td><td>The training objective: rebuild accurately, <i>minus</i> a
      penalty for drifting from the bell curve.</td><td><code>L = reconstruction + β·KL</code></td></tr>
    <tr><td><b>β-annealing</b></td><td>Start the KL penalty at zero and ramp it up.</td>
      <td>Fixes posterior collapse — see below</td></tr>
    <tr><td><b>Posterior collapse</b></td><td>A classic VAE failure: the penalty wins, the
      bottleneck stops carrying information, and every input rebuilds to the average.</td>
      <td><b>We hit this.</b> KL fell to 0.134 nats across 8 dimensions. Annealing fixed it →
      2.40</td></tr>
    </tbody></table>""")
    A("""<pre><b>ARCHITECTURE</b>
x (20 days x 9 features)
  --> LSTM(48) --> h --> Dense --> mu       (8 numbers)
                     \\-> Dense --> log s^2  (8 numbers)

  z = mu + s * eps        eps ~ N(0, I)     &lt;-- reparameterisation

  z --> RepeatVector(20) --> LSTM(48) --> TimeDistributed(Dense(9)) --> x_hat

<b>LOSS</b>   L = mean((x - x_hat)^2)  +  beta * KL
       KL = -0.5 * sum(1 + log s^2 - mu^2 - s^2)

<b>SCORE</b>  anomaly = reconstruction error on a NEW 20-day window</pre>""")
    A("""<div class="warn"><b>Why unsupervised?</b> We only have <b>5 labelled crises</b> in
    eight years of data. You cannot train and validate a normal classifier on 5 positive
    examples — any accuracy would be an accident of which event fell in which test split. So we
    never show the model a label. It learns only "what calm looks like". Labels are kept
    aside purely to <i>score</i> it afterwards.</div>""")

    A("<h3>Arm A — Convolutional Neural Network with transfer learning</h3>")
    A("""<div class="dl"><b>The core idea.</b> Take a network already trained on millions of
    ordinary photographs, and reuse its early layers — which have learned generic edge and blob
    detectors — on satellite radar images.</div>""")
    A("""<table><thead><tr><th>Concept</th><th>What it means</th><th>How we use it</th></tr></thead><tbody>
    <tr><td><b>Convolution</b></td><td>Slide a small filter across an image; each filter learns
      to fire on a particular local pattern (an edge, a corner, a bright blob).</td>
      <td>64×64 pixel patches cut around candidate ships</td></tr>
    <tr><td><b>Transfer learning</b></td><td>Reuse weights learned on a different, larger task
      instead of starting from random.</td><td>ImageNet → radar. Early layers transfer, deep
      "this is a dog" layers do not</td></tr>
    <tr><td><b>Depthwise-separable convolution</b></td><td>Split one expensive convolution into
      two cheap steps. Cost ratio <code>1/N + 1/D²</code> ≈ <b>0.115</b> — about 8.7× fewer
      multiplications.</td><td>Why we chose MobileNetV2: <b>this machine has no GPU</b></td></tr>
    <tr><td><b>Two-stage fine-tuning</b></td><td>First train only the new output layer with the
      backbone frozen; then unfreeze the top and train gently.</td>
      <td>A random new layer produces huge gradients that would wreck the pretrained filters</td></tr>
    <tr><td><b>Batch normalisation</b></td><td>Rescales activations using running averages
      collected during training.</td><td><b>Cost us 5 points.</b> ImageNet's averages are wrong
      for radar. Letting them adapt: 0.787 → 0.837</td></tr>
    <tr><td><b>Weak supervision</b></td><td>Train on imperfect, machine-generated labels instead
      of human ones.</td><td>A classical detector (CFAR) is the teacher. Detections on
      <i>land</i> are reliably not-ships</td></tr>
    <tr><td><b>Group-wise splitting</b></td><td>Split train/test by group, not by individual
      sample.</td><td>By satellite pass — patches from one image share sea state and the same
      ships</td></tr>
    </tbody></table>""")
    A("""<div class="bad"><b>Result: the CNN does not earn its place.</b> Test accuracy
    <b>0.837</b> against <b>0.946</b> for a trivial random forest on brightness statistics.
    The likely reason is structural, not a tuning problem: at ~15.6 metres per pixel a ship is
    <b>1–3 pixels</b>. It is a bright dot with no shape. Convolution exploits spatial structure,
    and here there is almost none — so simple brightness captures nearly all the information.</div>""")
    A("""<div class="say">"We report this as a negative result rather than hiding it. The
    project rule is that every component must beat a simple baseline, and this one doesn't. The
    honest conclusion is that 15-metre imagery is too coarse for shape-based ship detection —
    you would need ~1600-pixel chips, at three times the satellite quota."</div>""")

    A("<h3>Arm C — Transformer sentence embeddings, zero-shot</h3>")
    A("""<div class="dl"><b>The core idea.</b> Turn every headline into a 384-number vector such
    that sentences with similar <i>meaning</i> land near each other — even across languages.
    Then measure how close each headline sits to the idea of "a chokepoint is being
    disrupted".</div>""")
    A("""<table><thead><tr><th>Concept</th><th>What it means</th><th>How we use it</th></tr></thead><tbody>
    <tr><td><b>Transformer / attention</b></td><td>Each word looks at every other word and
      decides which ones matter for its meaning.</td><td>Inside MiniLM-L12, a 12-layer model</td></tr>
    <tr><td><b>Sentence embedding</b></td><td>One whole sentence compressed to a single vector
      of numbers.</td><td>384 dimensions, scaled to length 1</td></tr>
    <tr><td><b>Knowledge distillation</b></td><td>Train a small fast model to copy a big slow
      one.</td><td>MiniLM <i>is</i> distilled — that is why it runs on a CPU</td></tr>
    <tr><td><b>Cross-lingual alignment</b></td><td>Translations of the same sentence map to
      nearby vectors.</td><td><b>Measured: English↔Arabic cosine 0.866</b>, versus 0.073 for
      unrelated text</td></tr>
    <tr><td><b>Zero-shot classification</b></td><td>Classify without any training labels, by
      comparing to written descriptions of what you want.</td>
      <td>10 "disruption" sentences, 6 "ordinary shipping" sentences</td></tr>
    <tr><td><b>Contrastive scoring</b></td><td>Subtract the background similarity so you measure
      the signal, not the topic.</td><td><b>This is the crucial trick</b> — see below</td></tr>
    </tbody></table>""")
    A("""<pre><b>SCORING</b>
score(headline) = max cos(v, disruption_sentences)   "Iran closes the Strait of Hormuz"
                - max cos(v, calm_sentences)         "Oil prices steady in quiet trading"

<b>MEASURED</b>
  "Iran announces closure of the Strait of Hormuz"     +0.5165   HIGH
  Arabic translation of the same sentence              +0.4737   HIGH
  "Tanker struck by missile near Fujairah"             +0.4641   ELEVATED
  "DP World reports steady container throughput"       <b>-0.3043</b>   normal</pre>""")
    A("""<div class="good"><b>Why the subtraction matters.</b> Without it, any sentence about
    ports or oil scores moderately high, and the metric quietly becomes "how much shipping news
    happened today" — which we already count directly. Subtracting ordinary maritime language
    leaves only the escalation. Routine port news scoring <b>negative</b> is the proof it
    works.</div>""")

    A("<h3>Fusion and validation</h3>")
    A("""<table><thead><tr><th>Concept</th><th>What it means</th><th>How we use it</th></tr></thead><tbody>
    <tr><td><b>Ensemble fusion</b></td><td>Combine several models' outputs into one number.</td>
      <td>Weighted: Arm B 0.50, Arm C 0.50, <b>Arm A 0.00</b></td></tr>
    <tr><td><b>Causal percentile ranking</b></td><td>Rank today's value only against the past,
      never the whole dataset.</td><td>A full-sample rank would rank a 2019 day against 2026
      data that did not exist yet — that is leaking the future</td></tr>
    <tr><td><b>Leave-one-event-out (LOEO)</b></td><td>Hold out one event, use the rest, repeat.</td>
      <td>With 5 events this is the only split that guarantees every test fold contains
      something to detect</td></tr>
    <tr><td><b>Embargo</b></td><td>A gap between train and test so rolling averages cannot bleed
      across.</td><td>10 days</td></tr>
    <tr><td><b>Gradient boosting (GBM)</b></td><td>Many small decision trees, each fixing the
      previous one's mistakes.</td><td>Run only as a comparison — and it came out degenerate,
      which <i>is</i> the finding</td></tr>
    <tr><td><b>AUC / Mann-Whitney U</b></td><td>Probability a random positive scores above a
      random negative.</td><td>Arm C 0.848, Arm B 0.680; also used to test the SAR null</td></tr>
    </tbody></table>""")
    A("""<div class="warn"><b>Why the fusion is a rule and not a trained model.</b> A gradient
    boosting model fitted on 5 events is fitting noise. We ran it anyway to show this: labelling
    event windows made <b>233 of 282 rows positive (83%)</b>, so 2 of 3 folds had only one class
    and could not be scored at all. That degeneracy is the concrete demonstration that five
    labels cannot support a fitted combiner — so the shipped index uses weights taken from each
    arm's separately measured performance instead.</div>""")

    A("<h3>Classical baselines — every network must beat one</h3>")
    A("""<ul>
    <li><b>Random forest on brightness statistics — 0.946.</b> This beats our CNN, and we say so.</li>
    <li><b>CA-CFAR</b> — Constant False Alarm Rate: a classical radar detector that adapts its
    threshold to local sea clutter. Our vision baseline.</li>
    <li><b>Otsu thresholding</b> — automatic light/dark split, used to build the water mask.</li>
    </ul>
    <div class="note">The baseline is not just a comparison — it is a <b>correctness check</b>.
    When the CNN scored 35% and the baseline scored 94% on identical data, the gap proved the
    failure was in our model, not the task. It turned out one NaN pixel in 12,687 patches had
    turned every weight in the network to NaN.</div>""")

    A('<h3 id="notused">2.2 What we did NOT use</h3>')
    A("""<p>Listing this matters as much as the rest — an examiner will ask.</p>
    <table><thead><tr><th>Planned</th><th>Status</th><th>Why</th></tr></thead><tbody>
    <tr><td>xView3 vessel labels</td><td><span class="pill p-bad">not used</span></td>
      <td>Hundreds of GB. We used ImageNet weights + weak CFAR labels instead</td></tr>
    <tr><td>EfficientNet</td><td><span class="pill p-bad">not used</span></td>
      <td>No GPU — MobileNetV2 is ~8.7× cheaper per convolution</td></tr>
    <tr><td>Fine-tuned transformer</td><td><span class="pill p-bad">not done</span></td>
      <td>The encoder is <b>frozen</b>. Zero-shot only. Fine-tuning on 5 labels could not be validated</td></tr>
    <tr><td>Temporal Fusion Transformer</td><td><span class="pill p-bad">not built</span></td></tr>
    <tr><td>Claude embeddings</td><td><span class="pill p-bad">not used</span></td>
      <td>Used open sentence-transformers so the pipeline runs offline and free</td></tr>
    <tr><td>Kafka</td><td><span class="pill p-warn">built, not justified</span></td>
      <td>Measured 1.0 rows/sec against a broker's 1,000,000/sec design point = <b>0.0001%</b>.
      Built as a scale demonstration and labelled as such</td></tr>
    </tbody></table>""")

    # ══════════════════════════════ PART 3
    A('<h2>Part 3 · Tab-by-tab script</h2>')
    A("<p>What to say while showing each screen.</p>")

    tabs = [
      ("t1", "3.1 Overview", "01_overview",
       "The main screen. Risk index, backtest, live inference, and one card per arm.",
       """<p><b>Start at the top strip.</b> Four freshness cards. Only GDELT is genuinely live
       (0.6 hours old); market and satellite are ~1.3 days behind; the vessel layer says
       <b>no data</b>.</p>
       <div class="say">"Every panel shows its own age. This matters because a dashboard that
       renders all four identically implies a real-time view of the Gulf that does not exist.
       Satellite revisit is 1-6 days — 'live congestion' would be a lie."</div>
       <p><b>The risk index: 89.9 / 100, ELEVATED.</b> Note the amber box directly underneath —
       the caveat is not hidden in a tooltip, it sits next to the number.</p>
       <p><b>The backtest table</b> has three states, not two: DETECTED, missed, and
       <b>UNSCOREABLE</b>. Unscoreable means the event happened too early in our data for the
       system to have had enough history to predict it. Calling those "misses" would be unfair;
       calling them "detections" would be dishonest.</p>
       <div class="say">"We originally reported 3 of 5 detected. That was wrong — it credited us
       with catching the 2019 Gulf of Oman attack at zero days' lead, which isn't early warning,
       using history we didn't have. The honest number is 2 of 3 scoreable, median 36 days."</div>
       <p><b>Live inference is the demo.</b> Type any headline in any language and the backend
       runs the transformer on it. Note the Arabic line scores 0.4737 — within 0.04 of its
       English twin — while routine port news scores <b>negative</b>.</p>"""),

      ("t2", "3.2 Alerts", "02_alerts",
       "Alerts derived from measured state, not invented thresholds.",
       """<p>Six alerts. Notice that <b>data-quality alerts rank alongside risk alerts</b> —
       because a stale arm invalidates the index built on top of it.</p>
       <div class="say">"Two of these alerts are about our own models failing. Arm A contributes
       no signal; the CNN loses to a trivial baseline. A system that only alerts about the world
       and never about itself is one you cannot trust."</div>"""),

      ("t3", "3.3 Map", "03_map",
       "The 13 UAE ports, coloured by the coastline split that defines the thesis.",
       """<p><b>Red = inside the strait</b> (11 ports). <b>Green = bypass</b> (2 ports:
       Fujairah and Khor Fakkan, on the Gulf of Oman coast). Amber rings mark four ports whose
       coordinates are approximate.</p>
       <p>The whole hypothesis is here: if Hormuz is disrupted, traffic should reroute to the
       green ports. Fujairah is the world's #2 bunkering hub and sits outside the strait.</p>
       <div class="say">"This map shows the hypothesis, not a result. We measured Fujairah's
       vessel count across the 2026 crisis onset and it changed by plus zero point zero percent.
       The caveat under the map says exactly that — otherwise the map argues for a reroute our
       own data doesn't support."</div>"""),

      ("t4", "3.4 Satellite", "04_satellite",
       "Real Sentinel-1 radar imagery with CFAR detections circled.",
       """<p>449 chips across 4 ports and 4 crisis windows. Blue circles are detections the
       classical CFAR detector found on water — the vessel candidates.</p>
       <p><b>Why the image looks like this.</b> Radar backscatter is extremely skewed: calm
       water reads near zero, metal hulls read hundreds of times brighter. We convert to
       decibels and clip to a <b>fixed</b> range. Auto-stretching each image to its own contrast
       would make a busy anchorage and an empty sea look identical.</p>
       <div class="say">"The panel warns you automatically when a chip is partial — clipped by
       the satellite's swath edge. Such a chip reports fewer ships purely because of the
       footprint, and comparing it to a full chip would manufacture a congestion drop out of
       nothing. Same for Khalifa, where ascending and descending passes differ by 2x on
       identical water."</div>"""),

      ("t5", "3.5 Models &amp; DL", "05_models",
       "Every architecture with its equations, its numbers, and the mistake it taught us.",
       """<p>Four cards: the VAE, the CNN, the embeddings, and the fusion rule. Each shows the
       actual architecture, the measured results, and — in amber — the hard-won lesson.</p>
       <p><b>These are the lessons worth presenting</b>, because they are what a real project
       produces:</p>
       <ul>
       <li><b>β warm-up is load-bearing.</b> With the penalty on from step zero, the VAE's
       information content collapsed to 0.134 nats across 8 dimensions.</li>
       <li><b>One NaN pixel destroyed training.</b> <code>np.clip</code> passes NaN through; one
       bad gradient turned every weight to NaN; the model then output a constant and scored 35%.
       The trivial baseline is what exposed it.</li>
       <li><b>Frozen BatchNorm cost 5 accuracy points</b>, because ImageNet's statistics are
       wrong for radar.</li>
       <li><b>The GBM came out degenerate</b> — and that degeneracy is the proof that five
       labels cannot support a fitted model.</li>
       </ul>
       <div class="say">"The green box at the bottom is the point: these models are loaded into
       memory and executed per request. The backend runs deep learning live — it isn't serving
       numbers someone computed once and saved."</div>"""),

      ("t6", "3.6 Data pipeline", "06_pipeline",
       "How GDELT data actually arrives, and the 12-stage orchestration DAG.",
       """<p><b>The flow, left to right:</b> GDELT publishes every 15 minutes → our poller picks
       it up (offset 240 seconds, because publication lags the nominal slot) → we parse 61
       columns → apply a three-way Gulf filter → de-duplicate → write parquet.</p>
       <p><b>Two rules shown as amber boxes</b>, both learned the hard way:</p>
       <ul>
       <li><b>Index on publication time, never event time.</b> GDELT's "Day" field is when the
       event supposedly happened, and it can trail publication by <b>up to 365 days</b>. Using it
       would place information before anyone could have known it — textbook look-ahead bias.</li>
       <li><b>De-duplicate before counting.</b> GDELT emits one row per pair of actors mentioned,
       so one article becomes several rows — a measured <b>60–75% overcount</b>. A raw row count
       spikes when wire services repeat a story, not when something new happens.</li>
       </ul>
       <p><b>The DAG</b> shows all 12 pipeline stages and whether each is fresh or stale.
       Staleness is derived from file timestamps — if an input is newer than its output, the
       stage is stale. There is no separate database to drift out of sync with reality.</p>
       <div class="say">"Two stages are marked manual and never run automatically. Satellite
       collection costs about 5,947 processing units against a hard monthly cap of 30,000 where
       overage is refused, not billed. An orchestrator that helpfully re-collected because a
       timestamp moved would kill the month's quota."</div>"""),

      ("t7", "3.7 Semantics", "07_semantics",
       "25 encoded prohibitions — the rules the system will not let you break.",
       """<p>Every metric in the system has a definition, a provenance, caveats, and a list of
       <b>forbidden uses</b>. Each forbidden rule was learned by getting something wrong and
       measuring the cost.</p>
       <p>Examples: never rank crises on the raw Brent–WTI spread (its all-time maximum belongs
       to a US storage squeeze, not a Gulf crisis); never use raw GDELT row counts as a volume
       signal; never compare satellite passes of different look direction.</p>
       <div class="say">"This layer isn't documentation. These 25 rules are injected into the
       language model's prompt as guardrails when it writes the analyst brief — so the model
       physically cannot claim the satellite arm confirms disruption, because the registry
       records that it doesn't."</div>"""),
    ]

    for anchor, title, shot, cap, body in tabs:
        A(f'<h3 id="{anchor}">{title}</h3>')
        A(img(SHOTS / f"{shot}.png", cap))
        A(body)

    # ══════════════════════════════ PART 4
    A('<h2 id="limits">Part 4 · Honest limitations</h2>')
    A("""<p>State these before you are asked. They are the strongest part of the presentation,
    not the weakest.</p>
    <ol>
    <li><b>Five labelled events.</b> Everything else follows from this. It is why both models
    are unsupervised, why the fusion is a rule, and why we use leave-one-event-out.</li>
    <li><b>Arm B detects 1 of 5 events.</b> The 38-day lead is <i>one</i> event. It also fires on
    COVID and the Ukraine invasion — it detects market dislocation, not Gulf geography.</li>
    <li><b>Arm A found nothing.</b> Fujairah +0.0% across the onset, p = 0.26–0.89. We can rule
    out a large (&gt;33%) reroute, not a subtle one.</li>
    <li><b>The SAR CNN loses to a brightness baseline</b> (0.837 vs 0.946).</li>
    <li><b>No live vessel layer exists.</b> Measured, not assumed.</li>
    <li><b>Arm C's text is a degraded proxy.</b> We recover headlines from URL slugs; 39% of
    URLs are opaque numeric IDs, and those are missing-not-at-random — they skew by region and
    language.</li>
    <li><b>Satellite sampling density correlates with the label.</b> 2026 has ~2× the revisit of
    older windows <i>and</i> is a crisis window, so a model could learn "dense sampling ⇒ crisis"
    without ever seeing a future value.</li>
    </ol>""")
    A("""<div class="say">"If I had to defend one design decision, it's this: the risk endpoint
    returned HTTP 501 — not implemented — for most of the build, rather than a placeholder
    number. A placeholder is indistinguishable from a real index once it's on a screen, and it
    ends up quoted. It only started returning a value when there was a backtested model behind
    it."</div>""")

    A('<h2 id="run">4.2 How to run it</h2>')
    A("""<pre><b># everything: containers, API, model preloading, status report</b>
python -m backend.launch

<b># dashboard</b>        http://127.0.0.1:8000/
<b># API docs</b>         http://127.0.0.1:8000/docs

<b># individual pieces</b>
python -m backend.orchestration.dag --status     pipeline state
python -m backend.arms.market.vae                train the VAE
python -m backend.arms.sar.cnn                   train the CNN
python -m backend.fusion.combine                 fusion + backtest
python -m backend.features.splits                fold audit (validation gate)</pre>""")

    A("<p style='color:var(--dim);font-size:13px;margin-top:40px'>Generated from "
      "<code>docs/make_script.py</code>. Screenshots captured live from the running "
      "system; architecture rendered from <code>docs/make_architecture.py</code>.</p>")
    A("</div></body></html>")

    out = HERE / "PROJECT_SCRIPT.html"
    out.write_text("\n".join(H), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size/1024/1024:.1f} MB)")
