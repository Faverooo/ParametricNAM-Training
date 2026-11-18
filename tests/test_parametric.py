#!/usr/bin/env python3
"""Unit tests: W=0 equivalence, grad flow, bake exactness, FiLM bake, tiny synthetic train."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_fixture import legacy_dataset

from nam_param.parametric_wavenet import build_a1_parametric
from nam_param.variants import build_a1_parametric_variant
from nam_param.bake import verify_bake_equivalence, bake_knobs_into_nam, clone_parametric, fold_knobs_into_conv_bias
from nam_param.dataset import ParametricNAMDataset
from nam_param.losses_util import esr_loss, mse_loss
from nam.models.wavenet import WaveNet as NamWaveNet
from nam_param.a1_config import a1_standard_config


def test_zero_init_equals_stock():
    torch.manual_seed(0)
    param = build_a1_parametric()
    x = torch.randn(2, 8192)
    with torch.no_grad():
        y_stock = param.stock_forward(x, pad_start=True)
        y0 = param(x, torch.zeros(2, 2), pad_start=True)
        y1 = param(x, torch.ones(2, 2), pad_start=True)  # W=0,b=0 => identical
    d0 = (y_stock - y0).abs().max().item()
    d1 = (y_stock - y1).abs().max().item()
    d01 = (y0 - y1).abs().max().item()
    print(f"zero_init vs stock_forward: g=0->{d0:.3e} g=1->{d1:.3e} g0vs1->{d01:.3e}")
    assert d0 < 1e-5, d0
    assert d1 < 1e-5, d1
    assert d01 < 1e-5, d01


def test_gradients_flow_to_W():
    model = build_a1_parametric()
    x = torch.randn(2, 4096)
    g = torch.tensor([[0.3, 0.7], [0.9, 0.1]])
    y = model(x, g, pad_start=True)
    loss = y.pow(2).mean()
    loss.backward()
    grads = []
    for lin in model.knob_projections:
        assert lin.weight.grad is not None
        grads.append(float(lin.weight.grad.abs().sum().item()))
    total = sum(grads)
    print(f"grad flow to W: sum|grad|={total:.6g} over {len(grads)} layers")
    assert total > 0


def test_bake_exact():
    torch.manual_seed(1)
    model = build_a1_parametric()
    with torch.no_grad():
        for lin in model.knob_projections:
            lin.weight.normal_(0, 0.05)
            lin.bias.normal_(0, 0.01)
    x = torch.randn(2, 8192)
    g = torch.tensor([0.35, 0.8])
    diff = verify_bake_equivalence(model, g, x, rtol=1e-4, atol=1e-5)
    print(f"bake equivalence max abs diff={diff:.3e}")


def test_film_bake_exact():
    """FiLM fold: scale conv+mixer weights, scale bias, add shift — max abs diff < 1e-5."""
    torch.manual_seed(2)
    model = build_a1_parametric_variant(variant="film_linear", channels=(8, 4))
    with torch.no_grad():
        for proj in model.knob_projections:
            proj.scale.weight.normal_(0, 0.05)
            proj.shift.weight.normal_(0, 0.05)
            if proj.shift.bias is not None:
                proj.shift.bias.normal_(0, 0.01)
    x = torch.randn(2, 8192)
    g = torch.tensor([0.35, 0.8])
    diff = verify_bake_equivalence(model, g, x, rtol=1e-4, atol=1e-5)
    print(f"film bake equivalence max abs diff={diff:.3e}")
    assert diff < 1e-5, diff


def test_none_variant_ignores_g():
    model = build_a1_parametric_variant(variant="none", channels=(4, 2))
    x = torch.randn(1, 4096)
    with torch.no_grad():
        y0 = model(x, torch.zeros(1, 2), pad_start=True)
        y1 = model(x, torch.ones(1, 2), pad_start=True)
        ys = model.stock_forward(x, pad_start=True)
    assert (y0 - y1).abs().max().item() < 1e-6
    assert (y0 - ys).abs().max().item() < 1e-6
    print("none variant ignores g: OK")


def test_synthetic_trains():
    torch.manual_seed(0)
    ds_path = legacy_dataset()
    assert (ds_path / "manifest.json").exists()
    model = build_a1_parametric_variant(
        variant="shift_linear", channels=(4, 2), sample_rate=48000.0
    )
    ds = ParametricNAMDataset(
        ds_path, split="train", segment_samples=8000, receptive_field=model.receptive_field, num_segments=64, seed=0
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    model.train()
    for i in range(80):
        x, y, g = ds[i % len(ds)]
        x, y, g = x.unsqueeze(0), y.unsqueeze(0), g.unsqueeze(0)
        opt.zero_grad()
        pred = model(x, g, pad_start=False)
        n = min(pred.shape[-1], y.shape[-1])
        err = (pred[..., :n] - y[..., :n]).pow(2).mean()
        denom = y[..., :n].pow(2).mean().clamp_min(1e-8)
        loss = err / denom
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    print(f"synthetic train: loss {losses[0]:.4g} -> {losses[-1]:.4g} (min {min(losses):.4g})")
    assert losses[-1] < losses[0] * 0.9 or min(losses) < losses[0] * 0.5

    x = torch.from_numpy(ds.x_full[1000:9000].copy()).unsqueeze(0)
    with torch.no_grad():
        y0 = model(x, torch.tensor([[0.0, 0.5]]), pad_start=True)
        y1 = model(x, torch.tensor([[1.0, 0.5]]), pad_start=True)
    d = (y0 - y1).abs().max().item()
    print(f"knob response max|y(g=0)-y(g=1)|={d:.4g}")
    assert d > 1e-4


def test_nobias_still_expresses_Wg():
    model = build_a1_parametric_variant(
        variant="shift_linear", use_bias=False, channels=(4, 2)
    )
    for lin in model.knob_projections:
        assert lin.bias is None
        with torch.no_grad():
            lin.weight.normal_(0, 0.1)
    x = torch.randn(1, 4096)
    with torch.no_grad():
        y0 = model(x, torch.zeros(1, 2), pad_start=True)
        y1 = model(x, torch.ones(1, 2), pad_start=True)
    d = (y0 - y1).abs().max().item()
    print(f"nobias Wg effect maxdiff={d:.4g}")
    assert d > 1e-3


def test_validation_protocol():
    from train_param import validation_view, mean_validation_esr
    from nam_param.dataset import resolve_regions
    ds_path = legacy_dataset()
    model = build_a1_parametric_variant(variant="film_linear", channels=(4, 2))
    train = ParametricNAMDataset(ds_path, split="train", entries="train", num_segments=7)
    val = validation_view(train)
    assert val.entries == train.entries
    assert val.x_full is train.x_full and val.y_list is train.y_list
    assert val.region_start == train.region_stop
    assert val.region_stop == len(train.x_full)
    assert len(val) == len(train.y_list)  # every condition, regardless of num_segments
    # Default split=val must never silently select test entries.
    default_val = ParametricNAMDataset(ds_path, split="val")
    assert default_val.entries == train.entries
    fixed_train = ParametricNAMDataset(ds_path, split="train", fixed_g=[0.5, 0.5])
    fixed_val = validation_view(fixed_train)
    assert len(fixed_val) == 1
    assert fixed_val.entries[0]["g"] == [0.5, 0.5]
    e1, rows = mean_validation_esr(model, val, torch.device("cpu"))
    e2, rows2 = mean_validation_esr(model, val, torch.device("cpu"))
    assert rows == rows2 and e1 == e2
    assert len(rows) == len(train.entries)
    assert e1 == sum(r["esr_val_region"] for r in rows) / len(rows)
    assert all(r["start_sample"] == train.region_stop for r in rows)
    # Il vecchio helper a crop deve rispettare pre-roll e pesi per elemento,
    # anche quando l'ultimo batch ha dimensione diversa.
    from train_param import evaluate
    from torch.utils.data import DataLoader
    a = ParametricNAMDataset(ds_path, split="val", segment_samples=1000,
                             receptive_field=model.receptive_field, num_segments=9, seed=10)
    b = ParametricNAMDataset(ds_path, split="val", segment_samples=1000,
                             receptive_field=model.receptive_field, num_segments=9, seed=10)
    model.train()
    ea, ra = evaluate(model, DataLoader(a, batch_size=4), torch.device("cpu"))
    assert model.training
    eb, rb = evaluate(model, DataLoader(b, batch_size=3), torch.device("cpu"))
    assert abs(ea-eb) < 1e-5 and len(ra) == len(rb) == 9
    # Real NAM v3 split: no target overlap between train and val.
    assert resolve_regions(9_120_000) == (480_000, 8_688_000, 8_688_000, 9_120_000)
    print(f"validation: train-only, fixed-g, all {len(rows)} conditions, deterministic mean: OK")


def test_chunked_evaluation_alignment():
    from train_param import evaluate_region_esr, evaluate_per_file
    import numpy as np
    from types import SimpleNamespace
    torch.manual_seed(5)
    model = build_a1_parametric_variant(variant="film_linear", channels=(4, 2)).eval()
    x = torch.randn(18000)
    y = torch.randn(18000)
    g = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        pred = model(x.unsqueeze(0), g.unsqueeze(0), pad_start=True).squeeze(0).double()
    for start in (0, 123, 6000):
        expected = ((pred[start:] - y[start:].double()).square().sum()
                    / y[start:].double().square().sum()).item()
        actual = evaluate_region_esr(model, x, y, g, torch.device("cpu"),
                                     start=start, chunk_samples=3000)
        assert abs(expected - actual) < 1e-6, (start, expected, actual)
    ds = SimpleNamespace(y_list=[y.numpy()], x_full=x.numpy(), g_list=[g.numpy()],
                         entries=[{"y": "test.wav"}], val_start_abs=15000)
    full = evaluate_per_file(model, ds, torch.device("cpu"), chunk_samples=3000)[0]
    assert full["num_samples"] == 18000 and full["start_sample"] == 0
    assert "esr_full" in full and "esr_val_region" not in full
    print("chunked eval == whole forward: file start, RF pre-roll, chunk boundaries, all samples: OK")


def test_patched_forward_and_nam_loader():
    """Confronto indipendente: upstream non patchato, gating, head, pad e loader."""
    import tempfile, json
    from nam.models import init_from_nam
    from nam_param.parametric_wavenet import ParametricWaveNet, load_parametric
    from nam_param.variants import ParametricWaveNetVariant
    from nam_param.bake import clone_parametric
    max_diff = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for gated in (False, True):
            for head1x1 in (False, True):
                cfg = {"layers_configs": [{"input_size": 1, "condition_size": 1,
                    "channels": 4, "dilations": [1,2], "kernel_size": 3,
                    "activation": {"name":"PairMultiply", "primary":"Tanh", "secondary":"Sigmoid"} if gated else "Tanh",
                    "head_1x1_config": {"active": head1x1, "out_channels": 4},
                    "layer_1x1_config": {"active": not head1x1},
                    "head": {"out_channels": 1,"kernel_size":1,"bias":True}}],
                    "head_scale":0.02,"sample_rate":48000.0}
                stock = NamWaveNet.init_from_config(cfg)
                stock_export = stock._get_export_dict()
                x = torch.randn(2, 300)
                g = torch.tensor([[0.35, 0.8],[0.35,0.8]])
                for variant in ("none", "shift_linear", "shift_mlp", "film_linear"):
                    model = ParametricWaveNetVariant(init_from_nam(stock_export),
                        knob_names=["gain","treble"], variant=variant)
                    for pad in (True, False):
                        with torch.no_grad():
                            assert torch.equal(model(x,g,pad_start=pad), stock(x,pad_start=pad))
                    # Nonzero conditioning: verify both halves of a gate get folded.
                    with torch.no_grad():
                        for proj in model.knob_projections:
                            for param in proj.parameters(): param.normal_(0, 0.05)
                    path = Path(tmp)/"baked.nam"
                    export = bake_knobs_into_nam(model,g[0],path)
                    actual_json = json.loads(path.read_text())
                    loaded = init_from_nam(actual_json).eval()
                    assert export["version"] == stock_export["version"]
                    assert export["architecture"] == stock_export["architecture"]
                    assert export["config"] == stock_export["config"]
                    assert len(export["weights"]) == len(stock_export["weights"])
                    for pad in (True,False):
                        with torch.no_grad():
                            diff = (loaded(x,pad_start=pad)-model(x,g,pad_start=pad)).abs().max().item()
                        assert diff < 1e-6, (gated,head1x1,variant,diff)
                        max_diff=max(max_diff,diff)
                    checkpoint=Path(tmp)/"param.pt"
                    model.export_parametric(checkpoint)
                    restored=load_parametric(checkpoint)
                    with torch.no_grad():
                        assert torch.equal(model(x,g),restored(x,g))
                        assert torch.equal(model(x,g),clone_parametric(model)(x,g))
                # Classic wrapper also tested independently, not via stock_forward.
                classic=ParametricWaveNet(init_from_nam(stock_export),knob_names=["gain","treble"])
                with torch.no_grad(): assert torch.equal(classic(x,g),stock(x))
        # A1 standard JSON schema and exact expected parameter count.
        a1=build_a1_parametric_variant(variant="none")
        assert a1.receptive_field==4093
        assert sum(p.numel() for p in a1.parameters())==13801
        export=bake_knobs_into_nam(a1,[0.5,0.5],Path(tmp)/"a1.nam")
        upstream=NamWaveNet.init_from_config(a1_standard_config())._get_export_dict()
        assert export["config"]==upstream["config"]
        assert export["version"]==upstream["version"]
        assert len(export["weights"])==len(upstream["weights"])
    print(f"independent upstream forward + native .nam loader (all variants, gated/head, pad): max diff={max_diff:.3e}")


def test_warm_start_variants():
    import tempfile,json
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"stock.nam"
        donor=build_a1_parametric_variant(variant="none",channels=(4,2))
        path.write_text(json.dumps(donor.base._get_export_dict()))
        x=torch.randn(1,5000);g=torch.tensor([[0.2,0.7]])
        for v in ("none","shift_linear","shift_mlp","film_linear"):
            for bias in (False,True):
                model=build_a1_parametric_variant(variant=v,channels=(4,2),use_bias=bias)
                model.load_stock_nam_weights(path)
                with torch.no_grad(): assert torch.equal(model(x,g),donor(x,g))
    print("warm start: none/linear/nobias/MLP/FiLM: OK")


def test_dataset_causal_crops_and_wav():
    import tempfile, numpy as np
    from scipy.io import wavfile
    from nam_param.dataset import _read_wav_mono_f32, _write_wav_mono_f32_pcm24, resolve_regions
    ds=ParametricNAMDataset(legacy_dataset(), segment_samples=700, receptive_field=101,
                            num_segments=11, seed=7)
    assert len(ds)%len(ds.y_list)==0
    # Recover sampled position from deterministic RNG before __getitem__.
    rng=np.random.default_rng();rng.bit_generator.state=ds.rng.bit_generator.state
    start=int(rng.integers(ds.region_start,ds.region_stop-ds.segment_samples+1))
    x,y,g=ds[0]
    assert len(x)==len(y)+100
    expected=np.pad(ds.x_full[max(0,start-100):start+700], (max(0,100-start),0))
    np.testing.assert_array_equal(x.numpy(),expected)
    np.testing.assert_array_equal(y.numpy(),ds.y_list[0][start:start+700])
    short=ParametricNAMDataset(legacy_dataset(),segment_samples=10**9,receptive_field=101)
    assert short.segment_samples==short.region_len
    assert len(short[0][1])==short.region_len
    assert resolve_regions(912000)[1]>resolve_regions(912000)[0]
    assert resolve_regions(2)==(0,1,1,2)
    with tempfile.TemporaryDirectory() as tmp:
        audio=np.array([-1.,-.5,0.,.5,.999],dtype=np.float32)
        path=Path(tmp)/"pcm24.wav";_write_wav_mono_f32_pcm24(path,audio)
        out,sr=_read_wav_mono_f32(path)
        np.testing.assert_allclose(out,audio,atol=2e-7);assert sr==48000
        path=Path(tmp)/"float.wav";wavfile.write(path,48000,audio)
        np.testing.assert_array_equal(_read_wav_mono_f32(path)[0],audio)
        path=Path(tmp)/"pcm32.wav"
        values=np.array([-8,-1,0,1,8],dtype=np.int32);wavfile.write(path,48000,values)
        np.testing.assert_array_equal(_read_wav_mono_f32(path)[0],values.astype(np.float32)/2147483648.)
    print("causal crop RF/target alignment, small regions, PCM24/32 and float WAV: OK")


def test_loss_contracts():
    from train_param import compute_batch_loss
    from nam_param.losses_util import esr_per_item, preemphasized_esr_mean
    from nam.models.losses import esr,apply_pre_emphasis_filter
    from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss
    x=torch.tensor([[1.,2.,3.],[10.,20.,30.]])
    pred=torch.zeros_like(x)
    assert torch.allclose(esr(pred,x),esr_per_item(pred,x).mean())
    assert torch.equal(apply_pre_emphasis_filter(x,.85),x[:,1:]-.85*x[:,:-1])
    assert torch.isfinite(preemphasized_esr_mean(torch.zeros(2,10),torch.zeros(2,10)))
    for kind in ("mse","esr","esr_crop"):
        p=torch.randn(2,4096,requires_grad=True); y=torch.randn_like(p)
        loss=compute_batch_loss(p,y,.85,kind,2e-4,MultiResolutionSTFTLoss(),
                                file_energy=torch.tensor([1.,2.]))
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    print("loss: upstream ESR semantics, pre-emphasis, silent denominator, MSE/ESR+MRSTFT gradients: OK")


def test_file_normalized_loss_silent_crop():
    """Loss v2: un crop silenzioso non fa esplodere la loss (bug smoke test Colab)."""
    import numpy as np, tempfile, json, contextlib, io
    from train_param import compute_batch_loss, main, LOSS_DEF
    from nam_param.losses_util import file_train_energies, file_normalized_esr
    from nam.models.losses import apply_pre_emphasis_filter
    torch.manual_seed(0)
    # File "reale": 2 s di segnale, poi 1 s di rumore di fondo a -100 dBFS.
    sig=np.concatenate([np.random.default_rng(0).normal(0,.1,96000),
                        np.random.default_rng(1).normal(0,1e-5,48000)]).astype(np.float32)
    E=file_train_energies([sig],0,len(sig),.85)
    loud=torch.from_numpy(sig[:4800]); quiet=torch.from_numpy(sig[100000:104800])
    y=torch.stack([loud,loud,quiet]); pred=0.01*torch.randn_like(y)   # uscita piccola tipo init
    fe=E.repeat(3)
    old=compute_batch_loss(pred,y,.85,"esr_crop",0,None).item()
    new=compute_batch_loss(pred,y,.85,"esr",0,None,file_energy=fe).item()
    assert old>1e3, old            # v1: l'elemento silenzioso domina (ESR ~1e5)
    assert new<5, new              # v2: limitata, ~ ESR dei crop forti
    # Dentro un file = MSE pre-enfatizzato / costante (obiettivo NAM).
    mse=(apply_pre_emphasis_filter(pred,.85)-apply_pre_emphasis_filter(y,.85)).pow(2).mean()
    assert torch.allclose(torch.tensor(new),mse/E[0],rtol=1e-4)
    # Crop = intera regione: coincide con l'ESR pre-enfatizzato del file.
    full=torch.from_numpy(sig)[None]; pf=torch.zeros_like(full)
    assert abs(file_normalized_esr(pf,full,E,.85).item()-1.0)<1e-4
    # Energia nulla o shape errata: errore esplicito, non NaN silenziosi.
    for bad in (torch.zeros(3), torch.ones(2)):
        try: file_normalized_esr(pred,y,bad,.85); raise AssertionError("expected ValueError")
        except ValueError: pass
    # Il dataset restituisce l'indice file e il trainer rifiuta checkpoint v1.
    ds=ParametricNAMDataset(legacy_dataset(),segment_samples=500,num_segments=9,return_file_index=True)
    x,yy,g,fi=ds[4]; assert fi==4%len(ds.y_list) and len(yy)==500 and len(x)==500+ds.receptive_field-1
    with tempfile.TemporaryDirectory() as tmp:
        common=["--dataset",str(legacy_dataset()),"--variant","none","--channels","2,1",
                "--batch-size","2","--segment-samples","500","--val-every","2","--checkpoint-every","2",
                "--device","cpu","--out",tmp]
        with contextlib.redirect_stdout(io.StringIO()): main(common+["--steps","2"])
        st=torch.load(Path(tmp)/"train_state.pt",weights_only=False)
        assert st["args"]["loss_def"]==LOSS_DEF
        st["args"].pop("loss_def"); torch.save(st,Path(tmp)/"train_state.pt")   # simula checkpoint vecchio
        try:
            with contextlib.redirect_stdout(io.StringIO()): main(common+["--steps","4"])
            raise AssertionError("legacy checkpoint must not resume")
        except ValueError as e: assert "loss" in str(e)
    print(f"loss v2: silent crop old={old:.3g} new={new:.3g}; = MSE/E_file; full-file = ESR; legacy resume refused: OK")


def test_balanced_sampler_and_exact_resume():
    import tempfile,json,contextlib,io
    from train_param import StatefulSegmentSampler,main
    import train_param
    from unittest.mock import patch
    real_builder = train_param.build_model
    def checked_builder(args, knob_names=None):
        model = real_builder(args, knob_names)
        original = model.forward
        def checked_forward(*a, **kw):
            if torch.is_grad_enabled():
                assert model.training, "Validation left training in eval mode"
            return original(*a, **kw)
        model.forward = checked_forward
        return model
    ds=ParametricNAMDataset(legacy_dataset(),segment_samples=500,num_segments=40)
    sampler=StatefulSegmentSampler(ds,seed=2)
    it=iter(sampler);first=[next(it) for _ in range(9)]
    assert sorted(i%9 for i in first)==list(range(9))
    state=sampler.state_dict();rest=list(it)
    other=StatefulSegmentSampler(ds,seed=999);other.load_state_dict(state)
    assert rest==list(other)
    with tempfile.TemporaryDirectory() as tmp:
        common=["--dataset",str(legacy_dataset()),"--variant","film_linear",
            "--channels","2,1","--batch-size","2","--segment-samples","1000",
            "--segments-per-epoch","18","--val-every","3","--checkpoint-every","3",
            "--lr-step-every","2","--lr-gamma","0.9","--device","cpu","--seed","42"]
        full=Path(tmp)/"full";split=Path(tmp)/"split"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(train_param, "build_model", checked_builder):
            main(common+["--out",str(full),"--steps","12"])
            main(common+["--out",str(split),"--steps","6"])
            main(common+["--out",str(split),"--steps","12"])
        a=torch.load(full/"train_state.pt",weights_only=False)
        b=torch.load(split/"train_state.pt",weights_only=False)
        assert a["step"]==b["step"]==12
        assert a["epoch"]==b["epoch"]==1
        assert a["sched"]==b["sched"]
        assert abs(a["opt"]["param_groups"][0]["lr"]-.004*.9**6)<1e-12
        assert a["best_val"]==b["best_val"]
        for k in a["model_state_dict"]:
            assert torch.equal(a["model_state_dict"][k],b["model_state_dict"][k]),k
        for k,v in a["opt"]["state"].items():
            for name,t in v.items(): assert torch.equal(t,b["opt"]["state"][k][name])
        # Checkpoint già scritto allo step finale, poi val forzata: aggiornare
        # anche history/best nel train_state, non lasciarli alla versione vecchia.
        final=Path(tmp)/"final"
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(common+["--out",str(final),"--steps","3","--val-every","4"])
        state=torch.load(final/"train_state.pt",weights_only=False)
        assert state["best_val"]==result["best_val_esr"]
        assert state["history"][-1]["step"]==3

    print("balanced blocks + exact CPU resume across epoch boundary: weights/Adam/scheduler/best identical: OK")


def test_v3_loss_and_dc_contracts():
    from train_param import compute_batch_loss, loss_definition, LOSS_DEF, LOSS_DEF_V3
    torch.manual_seed(311)
    y = torch.randn(3, 1000) * .1
    p = (y + .01).detach().requires_grad_(True)
    pre_e = torch.tensor([.01, .02, .03])
    raw_e = torch.tensor([.04, .05, .06])
    v2 = compute_batch_loss(p, y, .85, "esr", 0, None, file_energy=pre_e)
    v3 = compute_batch_loss(p, y, .85, "esr_dual", 0, None, file_energy=pre_e,
                            raw_file_energy=raw_e, raw_weight=.7)
    dc = compute_batch_loss(p, y, .85, "esr_dc", 0, None, file_energy=pre_e,
                            raw_file_energy=raw_e, dc_weight=.7)
    expected = .7 * (((p-y)**2).mean(-1) / raw_e).mean()
    assert torch.allclose(v3-v2, expected, rtol=1e-5)
    assert torch.allclose(dc, v3, rtol=1e-5)  # constant error: MSE = squared mean
    v3.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    # Analytic DC attenuation with the same denominator.
    zero = torch.zeros(2, 1000); const = torch.ones_like(zero) * .01
    raw = compute_batch_loss(const, zero, None, "esr", 0, None, file_energy=torch.ones(2))
    pre = compute_batch_loss(const, zero, .85, "esr", 0, None, file_energy=torch.ones(2))
    assert abs(float(pre/raw)-.15**2) < 1e-7
    assert loss_definition("esr") == LOSS_DEF
    assert loss_definition("esr_dual") == LOSS_DEF_V3 != LOSS_DEF
    for bad in (None, torch.zeros(3), torch.ones(2)):
        try:
            compute_batch_loss(p,y,.85,"esr_dual",0,None,file_energy=pre_e,raw_file_energy=bad)
            raise AssertionError("must reject missing/invalid raw energy")
        except ValueError: pass
    # Quiet crop still has a fixed file denominator, not a crop denominator.
    quiet = torch.zeros(3, 1000)
    q = compute_batch_loss(torch.ones_like(quiet)*.01,quiet,.85,"esr_dual",0,None,
                           file_energy=pre_e,raw_file_energy=raw_e)
    assert torch.isfinite(q) and q < .1
    print("v3 exact formula, gradients, DC attenuation, silent crop, loss identity: OK")


def test_zero_input_diagnostics_and_summary():
    from train_param import evaluate_zero_input, compact_validation_summary
    model = build_a1_parametric_variant(variant="film_linear", channels=(2,1))
    model.train()
    rows = evaluate_zero_input(model, [[0.,0.],[.5,.5],[1.,1.]], torch.device("cpu"), measure_samples=200, batch_size=2)
    assert model.training and len(rows) == 3
    for r in rows:
        assert r["measure_samples"] == 200 and r["std"] < 1e-6
        assert abs(r["rms"]-abs(r["mean"])) < 1e-6
    fake = [{"g":[0.,0.],"esr_val_region":.3},{"g":[0.,1.],"esr_val_region":.5},
            {"g":[1.,0.],"esr_val_region":.1},{"g":[.5,.5],"esr_val_region":.2}]
    summary, text = compact_validation_summary(fake)
    assert summary["gain0_mean_esr"] == .4 and len(summary["worst3"]) == 3
    assert summary["worst3"][0]["esr_val_region"] == .5 and "gain0_mean=" in text
    print("zero-input pre-roll, mode preservation, condition summary: OK")


def _check_v3_resume_and_warmstart(loss_kind, expected_def):
    import tempfile, contextlib, io
    from train_param import main, LOSS_DEF_V3
    from nam_param.parametric_wavenet import load_parametric
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        source=build_a1_parametric_variant(variant="film_linear", channels=(2,1))
        source.export_parametric(root/"source.pt")
        common=["--dataset",str(legacy_dataset()),"--variant","film_linear","--channels","2,1",
                "--batch-size","2","--segment-samples","500","--segments-per-epoch","18",
                "--val-every","2","--checkpoint-every","2","--device","cpu","--seed","41",
                "--loss",loss_kind,"--grad-clip","1","--init-parametric",str(root/"source.pt")]
        full=root/"full";split=root/"split"
        with contextlib.redirect_stdout(io.StringIO()):
            report=main(common+["--out",str(full),"--steps","4"])
            main(common+["--out",str(split),"--steps","2"])
            main(common+["--out",str(split),"--steps","4"])
        a=torch.load(full/"train_state.pt",weights_only=False)
        b=torch.load(split/"train_state.pt",weights_only=False)
        assert a["args"]["loss_def"] == expected_def
        for k in a["model_state_dict"]: assert torch.equal(a["model_state_dict"][k],b["model_state_dict"][k]),k
        assert a["history"] == b["history"]
        assert report["zero_input_train_conditions"] and report["zero_input_test_conditions"]
        assert "per_condition" in report["history"][-1]
        for changed in (["--loss","esr"],["--raw-weight",".5"],["--grad-clip","0"]):
            try:
                with contextlib.redirect_stdout(io.StringIO()):main(common+["--out",str(split),"--steps","6"]+changed)
                raise AssertionError("mismatched objective/clip must not resume")
            except ValueError as e:assert "loss" in str(e) or "incompatible" in str(e)
        # Standard NAM bake remains valid after training with v3.
        m=load_parametric(full/"best.pt")
        diff=verify_bake_equivalence(m,torch.tensor([.25,.75]),torch.randn(1,5000),rtol=1e-4,atol=1e-5)
        assert diff < 1e-5
    print("v3 parametric warm-start, exact clipped resume, mismatch guards, standard bake: OK")


def test_v3_resume_and_warmstart():
    from train_param import LOSS_DEF_V3, LOSS_DEF_DC
    _check_v3_resume_and_warmstart("esr_dual", LOSS_DEF_V3)
    _check_v3_resume_and_warmstart("esr_dc", LOSS_DEF_DC)


if __name__ == "__main__":
    test_zero_init_equals_stock()
    test_gradients_flow_to_W()
    test_bake_exact()
    test_film_bake_exact()
    test_none_variant_ignores_g()
    test_nobias_still_expresses_Wg()
    test_synthetic_trains()
    test_validation_protocol()
    test_chunked_evaluation_alignment()
    test_patched_forward_and_nam_loader()
    test_warm_start_variants()
    test_dataset_causal_crops_and_wav()
    test_loss_contracts()
    test_file_normalized_loss_silent_crop()
    test_balanced_sampler_and_exact_resume()
    test_v3_loss_and_dc_contracts()
    test_zero_input_diagnostics_and_summary()
    test_v3_resume_and_warmstart()
    print("ALL TESTS PASSED")
