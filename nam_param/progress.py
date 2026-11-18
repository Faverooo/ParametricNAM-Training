"""progress.txt leggibile dal telefono (app Google Drive): stato del training in chiaro.

Sovrascritto (scrittura atomica) ogni N step, a ogni validazione, alla fine e in caso
di errore. Orari in Europe/Rome.
"""
from __future__ import annotations

import datetime as _dt
import os
import traceback
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Rome")
except Exception:  # tzdata mancante: ora locale del sistema
    _TZ = None


def now_rome() -> str:
    t = _dt.datetime.now(_TZ) if _TZ else _dt.datetime.now().astimezone()
    return t.strftime("%d/%m/%Y %H:%M:%S %Z")


def _fmt(v, spec=".6g"):
    return "n/d" if v is None else format(v, spec)


class ProgressWriter:
    def __init__(self, path, run_name="", total_steps=None):
        self.path = Path(path)
        self.f = {"run": run_name, "status": "AVVIO", "step": 0, "total": total_steps, "loss": None,
                  "val": None, "val_floored": None, "val_step": None, "best": None, "best_step": None,
                  "sps": None, "eta_h": None, "message": "", "started": now_rome()}

    def update(self, **kw):
        self.f.update(kw)

    def write(self, status=None, message=None):
        if status is not None:
            self.f["status"] = status
        if message is not None:
            self.f["message"] = message
        f = self.f
        total = f["total"]
        pct = f" ({100.0 * f['step'] / total:.1f}%)" if total else ""
        eta = "n/d" if f["eta_h"] is None else (f"{f['eta_h']:.2f} h" if f["eta_h"] >= 1 else f"{60 * f['eta_h']:.0f} min")
        lines = [
            f"STATO: {f['status']}",
            f"Run: {f['run']}",
            f"Aggiornato: {now_rome()}",
            f"Step: {f['step']}/{_fmt(total, 'd') if total else 'n/d'}{pct}",
            f"Loss train (media ultimi step): {_fmt(f['loss'])}",
            f"Ultima validazione (step {_fmt(f['val_step'], 'd') if f['val_step'] is not None else 'n/d'}): "
            f"ESR media {_fmt(f['val'])}, con pavimento {_fmt(f['val_floored'])}",
            f"Migliore (sceglie best.pt): {_fmt(f['best'])} allo step {_fmt(f['best_step'], 'd') if f['best_step'] is not None else 'n/d'}",
            f"Velocità: {_fmt(f['sps'], '.2f')} step/s",
            f"Tempo rimanente stimato: {eta}",
            f"Avviato/ripreso: {f['started']}",
        ]
        if f["message"]:
            lines.append(f"Messaggio: {f['message']}")
        lines.append("")
        lines.append("Se STATO è RUNNING ma 'Aggiornato' è vecchio di oltre 15 minuti, la sessione Colab è caduta:")
        lines.append("riapri il notebook e fai Runtime -> Esegui tutte (riprende da solo).")
        text = "\n".join(lines) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:  # Drive momentaneamente non scrivibile: non fermare il training
            print(f"(progress.txt non scritto: {e})")
        return text

    def error(self, exc: BaseException):
        if isinstance(exc, KeyboardInterrupt):
            return self.write("FERMATO", "interrotto a mano (stop della cella)")
        last = traceback.extract_tb(exc.__traceback__)[-1:] if exc.__traceback__ else []
        where = f" [{Path(last[0].filename).name}:{last[0].lineno}]" if last else ""
        return self.write("ERROR", f"{type(exc).__name__}: {exc}{where}")
