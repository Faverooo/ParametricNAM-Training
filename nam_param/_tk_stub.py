"""Minimal tkinter stub so nam.train can import on headless boxes without python3-tk."""
import sys
import types

try:
    import tkinter  # noqa: F401 — usarlo se disponibile, anche su una box headless
except ModuleNotFoundError as exc:
    if exc.name not in ("tkinter", "_tkinter"):
        raise
    tk = types.ModuleType("tkinter")
    tk.Tk = object
    tk.TclError = type("TclError", (Exception,), {})
    tk.StringVar = object
    tk.BooleanVar = object
    tk.IntVar = object
    tk.DoubleVar = object
    tk.END = "end"
    tk.DISABLED = "disabled"
    tk.NORMAL = "normal"
    tk.LEFT = "left"
    tk.RIGHT = "right"
    tk.TOP = "top"
    tk.BOTTOM = "bottom"
    tk.BOTH = "both"
    tk.X = "x"
    tk.Y = "y"
    tk.W = "w"
    tk.E = "e"
    tk.N = "n"
    tk.S = "s"
    tk.NSEW = "nsew"
    tk.HORIZONTAL = "horizontal"
    tk.VERTICAL = "vertical"
    tk.filedialog = types.ModuleType("tkinter.filedialog")
    tk.messagebox = types.ModuleType("tkinter.messagebox")
    tk.ttk = types.ModuleType("tkinter.ttk")
    tk.simpledialog = types.ModuleType("tkinter.simpledialog")
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.filedialog"] = tk.filedialog
    sys.modules["tkinter.messagebox"] = tk.messagebox
    sys.modules["tkinter.ttk"] = tk.ttk
    sys.modules["tkinter.simpledialog"] = tk.simpledialog
