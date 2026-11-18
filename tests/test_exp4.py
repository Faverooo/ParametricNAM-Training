#!/usr/bin/env python3
"""Test exp4 (4 knob): nomi file, sampler bilanciato su 81 condizioni, pavimento
del normalizzatore, warm-start espanso exp3 -> exp4, resume v3/v4, controlli Colab.

Gira su CPU in ~1-2 minuti. Uso: python tests/test_exp4.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_fixture import legacy_dataset

import nam_param._tk_stub  # noqa: F401,E402
from nam_param.dataset import ParametricNAMDataset, parse_condition_name, infer_knob_names  # noqa: E402
from nam_param.losses_util import floor_file_energies, floor_file_energies_paired, file_train_energies  # noqa: E402
from nam_param.variants import build_a1_parametric_variant  # noqa: E402
from nam_param.parametric_wavenet import load_parametric  # noqa: E402
from nam_param.warmstart import expand_knob_warm_start  # noqa: E402
import make_synthetic_4knob  # noqa: E402
import train_param  # noqa: E402
import colab_checks  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="exp4_tests_"))
_DS = None


def synthetic_ds():
    """81+8 file da 1 s, stessa struttura di dataset_4knob (creato una volta)."""
    global _DS
    if _DS is None:
        _DS = _TMP / "dataset_4knob"
        make_synthetic_4knob.make(_DS, seconds=1.0)
    return _DS


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def test_name_parsing():
    n, v = parse_condition_name("train/gain0.00_treble0.50_mid1.00_bass0.50.wav")
    assert n == ["gain", "treble", "mid", "bass"] and v == [0.0, 0.5, 1.0, 0.5]
    n, v = parse_condition_name("test\\gain0.15_treble0.40_mid0.65_bass0.85.wav")
    assert n == ["gain", "treble", "mid", "bass"] and v == [0.15, 0.4, 0.65, 0.85]
    assert parse_condition_name("train/gain0.00_treble1.00.wav") == (["gain", "treble"], [0.0, 1.0])
    for bad in ("train/gain_0.5.wav", "train/gain0.5_gain0.5.wav", "train/foo.wav"):
        try:
            parse_condition_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    m = json.loads((synthetic_ds() / "manifest.json").read_text())
    assert "knob_names" not in m and infer_knob_names(m) == ["gain", "treble", "mid", "bass"]
    # manifest reale (se presente sulla box), mai su Colab
    real = Path("/workspace/real4/manifest.json")
    if real.exists():
        rm = json.loads(real.read_text())
        assert infer_knob_names(rm) == ["gain", "treble", "mid", "bass"]
        assert len(rm["train"]) == 81 and len(rm["test"]) == 8
    # g che non corrisponde al nome -> errore
    bad = json.loads(json.dumps(m))
    bad["train"][5]["g"] = [0.0, 0.0, 0.0, 0.0] if bad["train"][5]["g"] != [0.0] * 4 else [1.0] * 4
    try:
        infer_knob_names(bad)
    except ValueError as e:
        assert "non corrisponde" in str(e)
    else:
        raise AssertionError("g/name mismatch not detected")
    # ordine dei knob diverso fra file -> errore
    bad = json.loads(json.dumps(m))
    bad["test"][0]["y"] = "test/treble0.25_gain0.25_mid0.25_bass0.25.wav"
    try:
        infer_knob_names(bad)
    except ValueError as e:
        assert "incoerente" in str(e)
    else:
        raise AssertionError("order mismatch not detected")
    # legacy 2 knob del pacchetto
    legacy = json.loads((legacy_dataset() / "manifest.json").read_text())
    assert infer_knob_names(legacy) == ["gain", "treble"]
    ds = quiet(ParametricNAMDataset, synthetic_ds(), segment_samples=1000, receptive_field=4093)
    assert ds.knob_names == ["gain", "treble", "mid", "bass"] and len(ds.y_list) == 81
    assert all(g.shape == (4,) for g in ds.g_list)
    print("name parsing / knob inference: OK")


def test_sampler_balance_81():
    ds = quiet(ParametricNAMDataset, synthetic_ds(), segment_samples=2000, receptive_field=4093,
               balanced=True, return_file_index=True, seed=3)
    n = len(ds.y_list)
    assert n == 81 and len(ds) % n == 0
    sampler = train_param.StatefulSegmentSampler(ds, seed=5)
    order = list(iter(sampler))
    files = [ds[i][3] for i in order]
    counts = Counter(files)
    assert set(counts) == set(range(81)) and len(set(counts.values())) == 1, counts
    for b in range(0, len(files), n):  # ogni blocco di 81 = tutte le condizioni una volta
        assert sorted(files[b:b + n]) == list(range(81))
    # batch reale da 12 sul DataLoader: su 27 batch (=4 blocchi) ogni condizione 4 volte
    sampler2 = train_param.StatefulSegmentSampler(ds, seed=7)
    loader = torch.utils.data.DataLoader(ds, batch_size=12, sampler=sampler2)
    seen = Counter()
    it = iter(loader)
    for _ in range(27):
        seen.update(next(it)[3].tolist())
    assert set(seen.values()) == {4}, seen
    # g del crop = g del file
    x, y, g, fi = ds[order[0]]
    assert torch.allclose(g, torch.from_numpy(ds.g_list[fi]))
    # resume del sampler: stessa sequenza
    s3 = train_param.StatefulSegmentSampler(ds, seed=11)
    it3 = iter(s3)
    first = [next(it3) for _ in range(40)]
    state = s3.state_dict()
    rest = list(it3)
    s4 = train_param.StatefulSegmentSampler(ds, seed=999)
    s4.load_state_dict(state)
    assert list(iter(s4)) == rest and first + rest == s3.order
    print(f"sampler balance over 81 conditions: OK ({len(order)} crops/epoch, {len(order)//81} per condition)")


def test_normalizer_floor():
    # tabella realistica: mediana -30 dB, un file a -89 dB, uno a -67 dB, uno a -54 dB
    db = [-89.0, -67.0, -54.0] + [-30.0] * 10 + [-20.0] * 5
    table = torch.tensor([10 ** (d / 10) for d in db], dtype=torch.float32)
    out, floor, idx = floor_file_energies(table, 30.0)
    assert math.isclose(10 * math.log10(floor), -60.0, abs_tol=1e-4), floor
    assert idx == [0, 1], idx  # -54 dB NON pavimentato
    assert torch.allclose(out[2:], table[2:]) and torch.allclose(out[:2], torch.full((2,), floor))
    same, f0, i0 = floor_file_energies(table, None)
    assert same is table and f0 is None and i0 == []
    for bad in (0.0, -3.0, float("nan")):
        try:
            floor_file_energies(table, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    # effetto sulla loss: offset DC a -70 dBFS sul file quasi muto
    T = 4800
    y = torch.zeros(2, T)
    y[0] = 10 ** (-89 / 20) * torch.randn(T)
    y[1] = 10 ** (-20 / 20) * torch.randn(T)
    pred = y + 10 ** (-70 / 20)
    e_pre = file_train_energies([y[0].numpy(), y[1].numpy()], 0, T, 0.85)
    e_raw = file_train_energies([y[0].numpy(), y[1].numpy()], 0, T, None)
    l3 = train_param.compute_batch_loss(pred, y, 0.85, "esr_dual", 0, None, e_pre, e_raw)
    # tabella di 11 file: i 2 del batch + 9 a -30 dB (mediana -30 dB -> pavimento -60 dB)
    raw11 = torch.cat([e_raw, torch.full((9,), 1e-3)])
    pre11 = torch.cat([e_pre, torch.full((9,), 4e-4)])
    rawf, pref, raw_floor, fidx = floor_file_energies_paired(raw11, pre11, 30.0)
    assert fidx == [0] and math.isclose(10 * math.log10(raw_floor), -60.0, abs_tol=1e-3)
    assert torch.allclose(pref[0] / rawf[0], pre11[0] / raw11[0])  # spettro del file conservato
    assert torch.equal(rawf[1:], raw11[1:]) and torch.equal(pref[1:], pre11[1:])
    l4 = train_param.compute_batch_loss(pred, y, 0.85, "esr_dual", 0, None, pref[:2], rawf[:2])
    assert l3.item() > 20 and l4.item() < 0.2, (l3.item(), l4.item())
    # definizione loss
    assert train_param.loss_definition("esr_dual", 30.0) == train_param.LOSS_DEF_V4
    assert train_param.loss_definition("esr_dual") == train_param.LOSS_DEF_V3
    try:
        train_param.loss_definition("esr", 30.0)
    except ValueError:
        pass
    else:
        raise AssertionError("floor with v2 must be refused")
    print(f"normalizer floor: OK (loss DC -70 dBFS on -89 dBFS file: v3={l3.item():.3g}, v4={l4.item():.3g})")


def _random_film(knob_names, seed):
    torch.manual_seed(seed)
    m = build_a1_parametric_variant(variant="film_linear", knob_dim=len(knob_names), knob_names=knob_names)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for proj in m.knob_projections:
            proj.scale.weight.normal_(0, 0.8)
            proj.shift.weight.normal_(0, 0.3)
            proj.shift.bias.normal_(0, 0.1)
    return m.eval()


def test_warm_start_expand_equivalence():
    src = _random_film(["gain", "treble"], 0)
    path = _TMP / "exp3_like" / "best.pt"
    src.export_parametric(path, metadata={"step": 1})
    src = load_parametric(path).eval()
    dst = build_a1_parametric_variant(variant="film_linear", knob_dim=4, knob_names=["gain", "treble", "mid", "bass"])
    info = expand_knob_warm_start(dst, src)
    dst.eval()
    assert info["new_knobs"] == ["mid", "bass"] and len(info["expanded"]) == 80, info
    torch.manual_seed(1)
    x = 0.3 * torch.randn(3, 4093 + 6000)
    worst = 0.0
    with torch.no_grad():
        for gt in ([0, 0], [0.5, 0.5], [1, 1], [0.25, 0.9]):
            g2 = torch.tensor([gt] * 3, dtype=torch.float32)
            y2 = src(x, g2, pad_start=False)
            for mb in ([0, 0], [0.5, 0.5], [1, 1], [0.3, 0.8]):
                g4 = torch.tensor([gt + mb] * 3, dtype=torch.float32)
                worst = max(worst, (dst(x, g4, pad_start=False) - y2).abs().max().item())
    assert worst < 1e-6, worst
    for proj in dst.knob_projections:
        assert proj.scale.weight[:, 2:].abs().max() == 0 and proj.shift.weight[:, 2:].abs().max() == 0
    # ordine diverso nella destinazione: abbinamento per nome
    dst2 = build_a1_parametric_variant(variant="film_linear", knob_dim=4, knob_names=["mid", "gain", "bass", "treble"])
    expand_knob_warm_start(dst2, src)
    dst2.eval()
    with torch.no_grad():
        g2 = torch.tensor([[0.7, 0.2]] * 3)
        d = (dst2(x, torch.tensor([[0.4, 0.7, 0.9, 0.2]] * 3), pad_start=False) - src(x, g2, pad_start=False)).abs().max().item()
    assert d < 1e-6, d
    # errori chiari
    for bad_names, variant in ((["gain", "mid", "bass", "presence"], "film_linear"),
                               (["gain", "treble", "mid", "bass"], "shift_linear")):
        b = build_a1_parametric_variant(variant=variant, knob_dim=4, knob_names=bad_names)
        try:
            expand_knob_warm_start(b, src)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_names, variant))
    # stesso controllo passando dal trainer (--init-parametric), passo 0 con LR ~0
    out = _TMP / "warm_main"
    quiet(train_param.main, ["--dataset", str(synthetic_ds()), "--out", str(out), "--variant", "film_linear",
                             "--channels", "16,8", "--steps", "1", "--batch-size", "2", "--segment-samples", "500",
                             "--lr", "1e-12", "--loss", "esr_dual", "--energy-floor-db", "30", "--device", "cpu",
                             "--val-every", "1", "--checkpoint-every", "1", "--init-parametric", str(path),
                             "--no-auto-resume"])
    trained = load_parametric(out / "last.pt").eval()
    rep = json.loads((out / "report.json").read_text())
    assert rep["warm_start"]["new_knobs"] == ["mid", "bass"]
    with torch.no_grad():
        g4 = torch.tensor([[0.25, 0.9, 0.6, 0.1]] * 3)
        d = (trained(x, g4, pad_start=False) - src(x, torch.tensor([[0.25, 0.9]] * 3), pad_start=False)).abs().max().item()
    assert d < 1e-6, d
    print(f"warm-start exp3->exp4 equivalence: OK (max diff {worst:.2e}, via trainer {d:.2e})")


def _common(out, steps, extra=()):
    return ["--dataset", str(synthetic_ds()), "--out", str(out), "--variant", "film_linear", "--channels", "16,8",
            "--steps", str(steps), "--batch-size", "3", "--segment-samples", "2400", "--lr", "1e-3",
            "--lr-gamma", "0.9", "--lr-step-every", "2", "--loss", "esr_dual", "--mrstft-weight", "2e-4",
            "--device", "cpu", "--val-every", "2", "--checkpoint-every", "2", "--seed", "4", *extra]


def test_resume_refusal_and_acceptance():
    fl = ["--energy-floor-db", "30"]
    # 1) run v3 (exp3) -> riprenderlo con v4 deve essere RIFIUTATO
    v3 = _TMP / "res_v3"
    quiet(train_param.main, _common(v3, 2))
    assert torch.load(v3 / "train_state.pt", weights_only=False)["args"]["loss_def"] == train_param.LOSS_DEF_V3
    try:
        quiet(train_param.main, _common(v3, 4, fl))
    except ValueError as e:
        assert "loss" in str(e)
    else:
        raise AssertionError("v3 train_state resumed with v4 loss")
    # 2) v4 interrotto e ripreso == v4 continuo
    a, b = _TMP / "res_a", _TMP / "res_b"
    quiet(train_param.main, _common(a, 6, fl))
    quiet(train_param.main, _common(b, 4, fl))
    quiet(train_param.main, _common(b, 6, fl))  # auto-resume da train_state.pt
    sa = torch.load(a / "train_state.pt", weights_only=False)
    sb = torch.load(b / "train_state.pt", weights_only=False)
    assert sa["step"] == sb["step"] == 6 and sa["args"]["loss_def"] == train_param.LOSS_DEF_V4
    for k in sa["model_state_dict"]:
        assert torch.equal(sa["model_state_dict"][k], sb["model_state_dict"][k]), k
    assert sa["history"] == sb["history"]
    # 2b) LR separato per FiLM (preset exp4): due gruppi Adam, resume esatto anche così
    fm = fl + ["--film-lr-mult", "10"]
    fa, fb = _TMP / "res_fa", _TMP / "res_fb"
    quiet(train_param.main, _common(fa, 4, fm))
    quiet(train_param.main, _common(fb, 2, fm))
    quiet(train_param.main, _common(fb, 4, fm))
    s_fa = torch.load(fa / "train_state.pt", weights_only=False)
    s_fb = torch.load(fb / "train_state.pt", weights_only=False)
    groups = s_fa["opt"]["param_groups"]
    assert len(groups) == 2 and math.isclose(groups[1]["lr"] / groups[0]["lr"], 10.0, rel_tol=1e-9), groups
    n_knob = sum(p.numel() for p in build_a1_parametric_variant(variant="film_linear", knob_dim=4).knob_projections.parameters())
    assert n_knob == 2160, n_knob  # 9 x 240 canali: U (4 col), W (4 col), b
    for k in s_fa["model_state_dict"]:
        assert torch.equal(s_fa["model_state_dict"][k], s_fb["model_state_dict"][k]), k
    # 2c) progress.txt leggibile: FINISHED, step, best e suo step, timestamp Europe/Rome
    txt = (a / "progress.txt").read_text()
    assert txt.startswith("STATO: FINISHED") and "Step: 6/6 (100.0%)" in txt, txt
    best_step = min(sa["history"], key=lambda h: h["val_esr_floored"])["step"]
    assert f"allo step {best_step}" in txt and "Aggiornato: " in txt and ("CET" in txt or "CEST" in txt), txt
    assert "step/s" in txt and "Ultima validazione (step 6)" in txt, txt
    # 2d) stesso dataset in un percorso diverso (nuova sessione Colab): ripresa accettata
    moved = _TMP / "moved_ds"
    if not moved.exists():
        shutil.copytree(synthetic_ds(), moved)
    args_moved = _common(b, 8, fl)
    args_moved[args_moved.index("--dataset") + 1] = str(moved)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        train_param.main(args_moved)
    assert "percorso nuovo" in log.getvalue() and "resumed at step=6" in log.getvalue()
    # 2e) errore durante il run -> progress.txt ERROR con messaggio
    e_dir = _TMP / "res_err"
    try:
        quiet(train_param.main, _common(e_dir, 2, fl + ["--init-parametric", str(_TMP / "non_esiste.pt")]))
    except Exception:
        pass
    else:
        raise AssertionError("missing init must fail")
    etxt = (e_dir / "progress.txt").read_text()
    assert etxt.startswith("STATO: ERROR") and "non_esiste.pt" in etxt, etxt
    # 3) cambiare il pavimento o il moltiplicatore LR in resume -> rifiutato
    for change in (["--energy-floor-db", "20"], fl + ["--film-lr-mult", "5"]):
        try:
            quiet(train_param.main, _common(b, 8, change))
        except ValueError as e:
            assert "incompatible" in str(e) or "loss" in str(e)
        else:
            raise AssertionError(change)
    # 4) checkpoint troncato (disconnessione durante la scrittura) -> usa train_state.prev.pt
    c = _TMP / "res_c"
    quiet(train_param.main, _common(c, 4, fl))
    assert (c / "train_state.prev.pt").exists()
    (c / "train_state.pt").write_bytes((c / "train_state.pt").read_bytes()[:1000])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        train_param.main(_common(c, 6, fl))
    assert "precedente" in log.getvalue() and torch.load(c / "train_state.pt", weights_only=False)["step"] == 6
    # 5) i soli pesi di un best v3 sono un warm-start valido per v4
    w = _TMP / "res_warm"
    quiet(train_param.main, _common(w, 2, fl + ["--init-parametric", str(v3 / "best.pt")]))
    rep = json.loads((w / "report.json").read_text())
    assert rep["loss_def"] == train_param.LOSS_DEF_V4 and rep["init_parametric"].endswith("best.pt")
    # report exp4
    assert len(rep["validation_per_file_esr"]) == 81 and len(rep["per_file_esr"]) == 8
    # nel toy amp, come nel tone stack reale, con treble=mid=bass=0 l'uscita quasi sparisce
    assert rep["floored_train_files"] == [f"train/gain{g}_treble0.00_mid0.00_bass0.00.wav"
                                          for g in ("0.00", "0.50", "1.00")], rep["floored_train_files"]
    assert set(rep["val_knob_marginals"]) == {"gain", "treble", "mid", "bass"}
    assert all("esr_floored" in r for r in rep["validation_per_file_esr"])
    assert (w / "val_per_condition.csv").read_text().count("\n") == 82
    assert (w / "test_per_file.csv").read_text().count("\n") == 9
    print("resume: v3->v4 refused, v4 exact resume, option changes refused, truncated state fallback, warm-start: OK")


def test_colab_checks():
    ds = synthetic_ds()
    good = colab_checks.verify_dataset(ds, quiet=True)
    assert good["train"] == 81 and good["test"] == 8
    work = _TMP / "checks"

    def broken(name, mutate):
        d = work / name / "dataset_4knob"
        shutil.copytree(ds, d)
        mutate(d)
        try:
            quiet(colab_checks.verify_dataset, d)
        except colab_checks.ControlloFallito as e:
            return str(e)
        raise AssertionError(f"{name}: problem not detected")

    t = "train/gain0.50_treble0.50_mid0.50_bass0.50.wav"
    assert "Mancano 1" in broken("missing", lambda d: (d / t).unlink())
    assert "caricamento non finito" in broken("partial", lambda d: (d / t).write_bytes((d / t).read_bytes()[:5000]))
    assert "input.wav non trovato" in broken("noinput", lambda d: (d / "input.wav").unlink())

    def sr44(d):
        from nam_param.dataset import _write_wav_mono_f32_pcm24, _read_wav_mono_f32
        y, _ = _read_wav_mono_f32(d / t)
        _write_wav_mono_f32_pcm24(d / t, y, sr=44100)
    assert "44100" in broken("sr", sr44)

    def two_knob(d):
        m = json.loads((d / "manifest.json").read_text())
        m["train"] = m["train"][:25]
        (d / "manifest.json").write_text(json.dumps(m))
    assert "attesi 81" in broken("count", two_knob)
    # dataset annidato / nome sbagliato / zip caricato al posto della cartella
    nested = work / "nested" / "dataset_4knob"
    shutil.copytree(ds, nested / "dataset_4knob")
    try:
        quiet(colab_checks.verify_dataset, nested)
    except colab_checks.ControlloFallito as e:
        assert "una cartella dentro l'altra" in str(e)
    else:
        raise AssertionError("nested")
    try:
        quiet(colab_checks.verify_dataset, work / "nowhere" / "dataset_4knob")
    except colab_checks.ControlloFallito as e:
        assert "non esiste" in str(e)
    # _cal e doppioni Drive: solo avvisi
    d = work / "warn" / "dataset_4knob"
    shutil.copytree(ds, d)
    (d / "_cal").mkdir()
    shutil.copy(d / t, d / "_cal" / "x.wav")
    shutil.copy(d / t, d / "train" / (Path(t).stem + " (1).wav"))
    info = quiet(colab_checks.verify_dataset, d)
    assert any("_cal" in w for w in info["warnings"]) and any("(1)" in w for w in info["warnings"])
    # input.wav accanto al dataset (in ParametricNAM/) va bene
    d2 = work / "alt" / "dataset_4knob"
    shutil.copytree(ds, d2)
    shutil.move(str(d2 / "input.wav"), str(d2.parent / "input.wav"))
    assert quiet(colab_checks.verify_dataset, d2, drive_root=d2.parent)["input"] == d2.parent / "input.wav"
    # hash input sbagliato
    try:
        quiet(colab_checks.verify_dataset, ds, input_sha256="0" * 64)
    except colab_checks.ControlloFallito as e:
        assert "hash" in str(e)
    else:
        raise AssertionError("input hash")
    # copia locale + SHA256, poi corruzione rilevata
    sums = {rel: colab_checks.sha256(ds / rel) for rel in ["input.wav", t]}
    loc = work / "local" / "dataset_4knob"
    quiet(colab_checks.copy_dataset_local, ds, loc, sums=sums)
    assert len(list((loc / "train").glob("*.wav"))) == 81
    bad_sums = dict(sums, **{t: "f" * 64})
    try:
        quiet(colab_checks.copy_dataset_local, ds, loc, sums=bad_sums)
    except colab_checks.ControlloFallito as e:
        assert "SHA256" in str(e) and not (loc / t).exists()
    else:
        raise AssertionError("sha")
    # zip del codice
    zdir = work / "zip"
    zdir.mkdir()
    import zipfile
    with zipfile.ZipFile(zdir / "old.zip", "w") as z:
        z.writestr("EXP4_VERSION.txt", "exp3")
    shutil.copy(zdir / "old.zip", zdir / "parametric_nam_code_exp4.zip")
    try:
        quiet(colab_checks.find_code_zip, zdir, "parametric_nam_code_exp4.zip")
    except colab_checks.ControlloFallito as e:
        assert "VERSIONE" in str(e)
    else:
        raise AssertionError("zip version")
    (zdir / "parametric_nam_code_exp4.zip").rename(zdir / "parametric_nam_code_exp4 (1).zip")
    try:
        quiet(colab_checks.find_code_zip, zdir, "parametric_nam_code_exp4.zip")
    except colab_checks.ControlloFallito as e:
        assert "(1)" in str(e)
    else:
        raise AssertionError("zip dup")
    print("Colab checks (missing/partial/44.1k/no input/count/nested/_cal/dup/hash/zip): OK")


if __name__ == "__main__":
    try:
        test_name_parsing()
        test_sampler_balance_81()
        test_normalizer_floor()
        test_warm_start_expand_equivalence()
        test_resume_refusal_and_acceptance()
        test_colab_checks()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print("ALL EXP4 TESTS PASSED")
