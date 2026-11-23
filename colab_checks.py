"""Run Colab preflight checks before installing dependencies.

Checks use the standard library and report likely causes and remedies in Italian.
Used by the Colab notebook and tests/test_exp4.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import wave
import zipfile
from pathlib import Path

CODE_VERSION = "exp4-1.0.0"
_TOKEN = re.compile(r"^([A-Za-z][A-Za-z]*?)(-?\d+(?:\.\d+)?)$")


class ControlloFallito(RuntimeError):
    """A readable error that the notebook shows before stopping training."""


def _fail(problems, title):
    text = "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, 1))
    raise ControlloFallito(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n{text}\n{'=' * 70}")


def _parse_name(rel):
    stem = Path(rel).stem
    out = []
    for tok in stem.split("_"):
        m = _TOKEN.match(tok)
        if not m:
            return None
        out.append((m.group(1).lower(), float(m.group(2))))
    return out


def _wav_info(path):
    with wave.open(str(path), "rb") as w:
        return {"channels": w.getnchannels(), "sampwidth": w.getsampwidth(),
                "rate": w.getframerate(), "frames": w.getnframes()}


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(path):
    sums = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            sums[rel.strip()] = h.lower()
    return sums


def find_code_zip(drive_root, name):
    """Find the code ZIP archive and verify that it contains the exp4 trainer."""
    drive_root = Path(drive_root)
    zp = drive_root / name
    stem = Path(name).stem
    dups = sorted(p.name for p in drive_root.glob(f"{stem}*.zip") if p.name != name) if drive_root.exists() else []
    if not zp.exists():
        hint = [f"Manca {zp}."]
        if dups:
            hint.append(f"Su Drive ci sono invece: {dups}. Rinomina quello giusto in '{name}' "
                        "(Drive aggiunge ' (1)' se carichi due volte lo stesso file).")
        if (drive_root / "parametric_nam_code.zip").exists():
            hint.append("C'è parametric_nam_code.zip: è lo zip di exp3, NON va bene per exp4. "
                        f"Carica colab_exp4\\{name} dal PC.")
        _fail(hint, "ZIP DEL CODICE NON TROVATO")
    if dups:
        print(f"ATTENZIONE: oltre a {name} ci sono {dups}. Uso {name}; cancella i duplicati per non confonderti.")
    try:
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            version = z.read("EXP4_VERSION.txt").decode().strip() if "EXP4_VERSION.txt" in z.namelist() else None
    except zipfile.BadZipFile:
        _fail([f"{zp} è danneggiato o ancora in caricamento. Aspetta la fine del caricamento su Drive o ricaricalo."],
              "ZIP DEL CODICE ROTTO")
    if bad is not None:
        _fail([f"File corrotto nello zip: {bad}. Ricarica lo zip."], "ZIP DEL CODICE ROTTO")
    if version != CODE_VERSION:
        _fail([f"Lo zip contiene la versione {version!r}, il notebook si aspetta {CODE_VERSION!r}. "
               "Hai caricato uno zip vecchio o il notebook vecchio: usa entrambi da colab_exp4\\."],
              "VERSIONE DEL CODICE SBAGLIATA")
    return zp


def verify_dataset(dataset_dir, expected_train=81, expected_test=8, expected_knobs=("gain", "treble", "mid", "bass"),
                   input_sha256=None, drive_root=None, sample_rate=48000, sampwidth=3, quiet=False):
    """Validate the dataset directory, manifest, input.wav, and all WAV headers and sizes.

    Return the paths and counts, or raise ControlloFallito with all detected issues.
    Audio samples are not decoded.
    """
    ds = Path(dataset_dir)
    problems, warnings_ = [], []
    if not ds.exists():
        hints = [f"La cartella {ds} non esiste su Drive."]
        parent = ds.parent
        if parent.exists():
            similar = sorted(p.name for p in parent.iterdir() if "4knob" in p.name.lower() or p.name.lower().startswith("dataset"))
            if similar:
                hints.append(f"In {parent} ci sono: {similar}. Il nome deve essere esattamente '{ds.name}'.")
            if (parent / (ds.name + ".zip")).exists():
                hints.append(f"Hai caricato {ds.name}.zip: Drive non lo scompatta. Carica la CARTELLA, non lo zip.")
        else:
            hints.append(f"Non esiste nemmeno {parent}: controlla DRIVE_ROOT_NAME nella cella di configurazione "
                         "e che Drive sia montato (cartella 'Il mio Drive' = /content/drive/MyDrive).")
        _fail(hints, "DATASET NON TROVATO")
    if not (ds / "manifest.json").exists():
        nested = [p for p in ds.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        hint = [f"Manca {ds / 'manifest.json'}."]
        if nested:
            hint.append(f"Il manifest è in {nested[0]}: hai una cartella dentro l'altra. Sposta il contenuto "
                        f"un livello sopra o imposta DATASET_DIR_NAME = '{ds.name}/{nested[0].name}'.")
        _fail(hint, "MANIFEST MANCANTE")
    try:
        manifest = json.loads((ds / "manifest.json").read_text())
    except Exception as e:
        _fail([f"manifest.json illeggibile ({e}): forse ancora in caricamento. Ricaricalo dal PC."], "MANIFEST ROTTO")
    train, test = manifest.get("train", []), manifest.get("test", [])
    if len(train) != expected_train:
        problems.append(f"Il manifest ha {len(train)} file train, attesi {expected_train}. È il manifest di un altro dataset "
                        "(es. quello a 2 knob, 25 file)?")
    if len(test) != expected_test:
        problems.append(f"Il manifest ha {len(test)} file test, attesi {expected_test}.")
    if int(manifest.get("sample_rate", 0)) != sample_rate:
        problems.append(f"manifest sample_rate={manifest.get('sample_rate')}, atteso {sample_rate}.")
    for e in train + test:
        parsed = _parse_name(e.get("y", ""))
        names = tuple(n for n, _ in parsed) if parsed else None
        if names != tuple(expected_knobs):
            problems.append(f"Nome file {e.get('y')} non nel formato {'_'.join(k + 'X.XX' for k in expected_knobs)}.wav")
            break
        if len(e["g"]) != len(expected_knobs) or any(abs(a - b) > 1e-3 for a, b in zip(e["g"], (v for _, v in parsed))):
            problems.append(f"g={e['g']} non corrisponde al nome {e['y']}: manifest modificato a mano?")
            break
    # input.wav can be inside the dataset or alongside it in the ParametricNAM folder.
    x_name = manifest.get("x", "input.wav")
    cands = [ds / x_name] + ([Path(drive_root) / x_name] if drive_root else []) + [ds.parent / x_name]
    x_path = next((c for c in cands if c.exists()), None)
    x_info = None
    if x_path is None:
        problems.append(f"input.wav non trovato (cercato in {', '.join(str(c.parent) for c in cands)}). "
                        "Caricalo dentro dataset_4knob/ (c'è già sul PC in dataset_4knob\\input.wav).")
    else:
        try:
            x_info = _wav_info(x_path)
        except Exception as e:
            problems.append(f"{x_path} non è un WAV leggibile ({e}): ancora in caricamento?")
        if x_info and (x_info["rate"] != sample_rate or x_info["channels"] != 1):
            problems.append(f"input.wav: {x_info['rate']} Hz, {x_info['channels']} canali; atteso {sample_rate} Hz mono.")
        if x_info and input_sha256:
            h = sha256(x_path)
            if h != input_sha256.lower():
                problems.append(f"input.wav ha hash {h[:12]}…, atteso {input_sha256[:12]}…: non è il file NAM v3 usato "
                                "per il render (o è troncato). Ricarica dataset_4knob\\input.wav dal PC.")
    expected_bytes = x_path.stat().st_size if x_path is not None else None
    missing, wrong = [], []
    for e in train + test:
        p = ds / e["y"]
        if not p.exists():
            missing.append(e["y"])
            continue
        size = p.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            wrong.append(f"{e['y']}: {size} byte invece di {expected_bytes} (caricamento non finito o file troncato)")
            continue
        try:
            info = _wav_info(p)
        except Exception as ex:
            wrong.append(f"{e['y']}: WAV illeggibile ({ex})")
            continue
        if info["rate"] != sample_rate:
            wrong.append(f"{e['y']}: sample rate {info['rate']} Hz invece di {sample_rate} (render sbagliato)")
        elif info["channels"] != 1:
            wrong.append(f"{e['y']}: {info['channels']} canali invece di mono")
        elif info["sampwidth"] != sampwidth:
            wrong.append(f"{e['y']}: {8 * info['sampwidth']} bit invece di {8 * sampwidth}")
        elif x_info and info["frames"] != x_info["frames"]:
            wrong.append(f"{e['y']}: {info['frames']} campioni invece di {x_info['frames']} (non allineato a input.wav)")
    if missing:
        problems.append(f"Mancano {len(missing)} file su {len(train) + len(test)}: {missing[:5]}{' …' if len(missing) > 5 else ''}. "
                        "Se hai appena caricato, aspetta che Drive finisca e rilancia questa cella.")
    if wrong:
        problems.append(f"{len(wrong)} file con problemi: " + "; ".join(wrong[:5]) + (" …" if len(wrong) > 5 else ""))
    listed = {e["y"] for e in train + test}
    extra = sorted(f"{sub}/{p.name}" for sub in ("train", "test") if (ds / sub).exists()
                   for p in (ds / sub).iterdir() if f"{sub}/{p.name}" not in listed)
    if extra:
        warnings_.append(f"{len(extra)} file in train/test non elencati nel manifest (ignorati): {extra[:5]}. "
                         "Se finiscono con ' (1).wav' sono doppioni di Drive: cancellali.")
    cal = ds / "_cal"
    if cal.exists():
        n = sum(1 for _ in cal.rglob("*.wav"))
        warnings_.append(f"C'è la cartella _cal/ ({n} wav): NON serve al training ed è ignorata. "
                         "Puoi cancellarla da Drive per risparmiare spazio.")
    if problems:
        _fail(problems, "IL DATASET SU DRIVE NON È PRONTO — training NON avviato")
    for w in warnings_:
        print("ATTENZIONE:", w)
    if not quiet:
        print(f"Dataset OK: {len(train)} train + {len(test)} test, {x_info['frames']} campioni "
              f"({x_info['frames'] / sample_rate:.1f} s) a {sample_rate} Hz, input = {x_path}")
    return {"dataset": ds, "input": x_path, "train": len(train), "test": len(test),
            "frames": x_info["frames"], "bytes_per_file": expected_bytes, "warnings": warnings_}


def copy_dataset_local(src, dst, input_path=None, sums=None):
    """Copy the manifest and its WAV files to Colab's local disk.

    Skip files that were already copied with the same size. If sums is provided,
    verify SHA-256 after copying to catch truncated downloads from Drive.
    """
    src, dst = Path(src), Path(dst)
    manifest = json.loads((src / "manifest.json").read_text())
    rels = [e["y"] for e in manifest["train"] + manifest["test"]]
    x_name = manifest.get("x", "input.wav")
    jobs = [(src / "manifest.json", dst / "manifest.json"),
            (Path(input_path) if input_path else src / x_name, dst / x_name)] + [(src / r, dst / r) for r in rels]
    need = sum(a.stat().st_size for a, _ in jobs)
    free = shutil.disk_usage(dst.parent if dst.parent.exists() else "/").free
    if free < need * 1.2:
        _fail([f"Disco locale Colab: liberi {free / 1e9:.1f} GB, servono {need / 1e9:.1f} GB."], "SPAZIO DISCO INSUFFICIENTE")
    t0 = time.time()
    copied = 0
    for i, (a, b) in enumerate(jobs):
        b.parent.mkdir(parents=True, exist_ok=True)
        if b.exists() and b.stat().st_size == a.stat().st_size and b.name != "manifest.json":
            continue
        tmp = b.with_name(b.name + ".part")
        shutil.copyfile(a, tmp)
        os.replace(tmp, b)
        copied += 1
        if copied % 10 == 0:
            print(f"  copiati {copied} file ({time.time() - t0:.0f} s)")
    print(f"Dataset locale pronto in {dst} ({copied} copiati, {len(jobs) - copied} già presenti, {time.time() - t0:.0f} s)")
    if sums:
        bad = []
        for rel, h in sums.items():
            p = dst / rel
            if p.exists() and sha256(p) != h:
                bad.append(rel)
            elif not p.exists():
                bad.append(rel + " (manca)")
        if bad:
            for rel in bad:
                q = dst / rel.split(" ")[0]
                if q.exists():
                    q.unlink()  # Let the next attempt copy it again.
            _fail([f"{len(bad)} file con SHA256 diverso dal PC: {bad[:5]}. Sono corrotti o diversi su Drive: "
                   "ricaricali dal PC (cartella dataset_4knob), poi rilancia. Se hai rigenerato il dataset di "
                   "proposito, imposta VERIFY_SHA256 = False."], "DATASET DIVERSO DA QUELLO DEL PC")
        print(f"SHA256 OK per {len(sums)} file")
    return dst


def check_ram(need_gb):
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1e6
                if avail < need_gb:
                    _fail([f"RAM disponibile {avail:.1f} GB, ne servono ~{need_gb:.1f} per tenere il dataset in memoria. "
                           "Riavvia il runtime (Runtime -> Riavvia sessione) per liberare la RAM."], "RAM INSUFFICIENTE")
                return avail
    except FileNotFoundError:
        return None
