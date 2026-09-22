"""
Desktop GUI for Mojito L1 data processing.

Lets you assemble a data stream by adding and subtracting Mojito L1 packages
(noise, MBHB, GB, EMRI, SOBHB), tune every ``process_pipeline`` parameter, and
inspect the processed time series and amplitude spectral density.

Run with::

    mojito-barkeeper-gui

or::

    python -m mojito_barkeeper.gui
"""

from __future__ import annotations

import argparse
import importlib.resources
import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk  # noqa: E402
from tkinter import filedialog, font as tkfont, messagebox, ttk  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from mojito_barkeeper.lab import (  # noqa: E402
    AET_CHANNELS,
    CADENCE_CHOICES,
    TIME_UNITS,
    UI_SCALE_CHOICES,
    XYZ_CHANNELS,
    DataPackage,
    GuiState,
    PipelineError,
    PipelineSettings,
    TimeWindow,
    amplitude_spectral_density,
    clamp_ui_scale,
    discover_packages,
    is_saved_pipeline_output,
    load_gui_state,
    process,
    save_gui_state,
    save_result,
    SaveOptions,
    suggest_ui_scale,
)

CHANNEL_SETS = {"XYZ": XYZ_CHANNELS, "AET": AET_CHANNELS}
WINDOW_TYPES = ("tukey", "hann", "hamming", "blackman", "planck")
NATIVE_CADENCE = "native (no downsampling)"
CADENCE_LABELS = tuple(
    NATIVE_CADENCE if cadence is None else f"{cadence:g}" for cadence in CADENCE_CHOICES
)
MAX_PLOT_POINTS = 10**7
BYTES_PER_SAMPLE = 8

#: Mojito-themed palette derived from the app logo.
THEME = {
    "bg": "#f4faf6",
    "surface": "#ffffff",
    "header": "#1a4d32",
    "accent": "#2d8b57",
    "accent_hover": "#256f46",
    "accent_light": "#d8f0e2",
    "accent_muted": "#a0c4b0",
    "text": "#1a3024",
    "text_muted": "#5a7568",
    "text_on_dark": "#ffffff",
    "text_subtle_on_dark": "#c8e6d4",
    "border": "#b8d4c4",
    "error": "#b5453a",
    "plot_total": "#1a4d32",
    "plot_grid": "#c5ddd0",
}
#: Colours for the individual contributions; the total uses ``plot_total``.
COMPONENT_COLORS = (
    "#3d9970", "#e67e22", "#3498db", "#9b59b6", "#16a085",
    "#d35400", "#8e44ad", "#27ae60", "#2980b9",
)
#: L1 sampling rate assumed only for the memory estimate shown before loading.
NOMINAL_SOURCE_FS = 0.4

#: Tk named fonts restyled when the display scale changes.
SCALED_FONTS = (
    "TkDefaultFont",
    "TkTextFont",
    "TkFixedFont",
    "TkMenuFont",
    "TkHeadingFont",
    "TkCaptionFont",
    "TkSmallCaptionFont",
    "TkIconFont",
    "TkTooltipFont",
)

#: Package-table column widths at scale 1.0, in pixels.
TREE_COLUMN_WIDTHS = {"use": 44, "op": 36, "package": 300, "path": 420}

BASE_WINDOW_SIZE = (1200, 780)
BASE_MIN_SIZE = (820, 560)
BASE_TREE_ROW_HEIGHT = 18
BASE_FIGURE_DPI = 100
#: DPI used when exporting plots via the matplotlib toolbar (screen stays at BASE_FIGURE_DPI).
SAVE_FIGURE_DPI = 300
BASE_TOOLBAR_HEIGHT = 28
#: Section headings, relative to the body font.
HEADING_FONT_RATIO = 1.05
BASE_HINT_WRAPLENGTH = 200
BASE_PROGRESS_LENGTH = 140
BASE_LOGO_SIZE = 30
BASE_HEADER_TITLE_SIZE = 12
BASE_HEADER_SUBTITLE_SIZE = 8


matplotlib.rcParams["savefig.dpi"] = SAVE_FIGURE_DPI
matplotlib.rcParams["savefig.bbox"] = "tight"


def logo_path() -> Path:
    """Return the bundled logo, falling back to a repo checkout for dev runs."""
    try:
        return Path(importlib.resources.files("mojito_barkeeper") / "assets" / "mojito_barkeeper_logo.png")
    except (ModuleNotFoundError, TypeError, FileNotFoundError):
        return Path(__file__).resolve().parents[2] / "assets" / "mojito_barkeeper_logo.png"


def decimate_for_display(series: np.ndarray, max_points: int = MAX_PLOT_POINTS):
    """Subsample a long series so matplotlib stays responsive."""
    if len(series) <= max_points:
        return np.arange(len(series)), series
    step = int(np.ceil(len(series) / max_points))
    index = np.arange(0, len(series), step)
    return index, series[index]


class MojitoPipelineGUI(tk.Tk):
    """Main application window."""

    def __init__(
        self,
        data_root: str | None = None,
        dt: float = 10.0,
        ui_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.title("Mojito Barkeeper")
        self.configure(bg=THEME["bg"])

        self.packages: list[DataPackage] = []
        self.result = None
        self._worker: threading.Thread | None = None
        self._messages: queue.Queue = queue.Queue()
        self._drain_job: str | None = None
        self._logo_photo: ImageTk.PhotoImage | None = None
        self._logo_source = Image.open(logo_path()).convert("RGBA")

        stored = load_gui_state()
        # None means "pick up where the last session left off"; an explicit
        # empty string starts with nothing selected.
        if data_root is None:
            data_root = stored.data_root

        # Explicit request wins, then the stored preference, then a guess from
        # the screen size so HiDPI panels are not unusably small on first run.
        self._ui_scale = clamp_ui_scale(
            ui_scale
            if ui_scale is not None
            else stored.ui_scale
            if stored.ui_scale is not None
            else suggest_ui_scale(self.winfo_screenwidth(), self.winfo_screenheight())
        )
        self._base_font_sizes = {
            name: tkfont.nametofont(name).cget("size") for name in SCALED_FONTS
        }
        self._style = ttk.Style(self)
        self._apply_theme()
        # Widgets whose size is set in pixels rather than font units, so they
        # have to be rescaled by hand.
        self._section_toggles: list[ttk.Checkbutton] = []
        self._toolbars: list[NavigationToolbar2Tk] = []

        # Tk scaling has to be in place before the matplotlib toolbars are
        # built, because they size their icons from it once at construction.
        self._set_tk_scaling(self._ui_scale)

        self._build_variables(data_root, dt, stored)
        self._build_layout()
        self._reset_settings()
        self._bind_traces()
        self.apply_ui_scale(self._ui_scale, remember=False)
        if self.var_data_root.get():
            self.scan_data_root()
        else:
            self.var_status.set("Choose a data directory with 'Browse…' to list packages")
        self._drain_job = self.after(100, self._drain_messages)

    # -------------------------------------------------------------- display

    @property
    def ui_scale(self) -> float:
        return self._ui_scale

    def _set_tk_scaling(self, scale: float) -> None:
        """Tell Tk how many pixels make up a point at this display scale."""
        self.tk.call("tk", "scaling", scale * 96.0 / 72.0)

    def _apply_theme(self) -> None:
        """Apply the Mojito colour palette to ttk widgets."""
        style = self._style
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(".", background=THEME["bg"], foreground=THEME["text"])
        style.configure("TFrame", background=THEME["bg"])
        style.configure(
            "Header.TFrame",
            background=THEME["header"],
        )
        style.configure(
            "TLabelframe",
            background=THEME["bg"],
            bordercolor=THEME["border"],
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=THEME["bg"],
            foreground=THEME["header"],
            font=("TkDefaultFont", 10, "bold"),
        )
        style.configure(
            "TLabel",
            background=THEME["bg"],
            foreground=THEME["text"],
        )
        style.configure(
            "Muted.TLabel",
            background=THEME["bg"],
            foreground=THEME["text_muted"],
        )
        style.configure(
            "Error.TLabel",
            background=THEME["bg"],
            foreground=THEME["error"],
        )
        style.configure(
            "TButton",
            background=THEME["accent"],
            foreground=THEME["text_on_dark"],
            bordercolor=THEME["accent"],
            focuscolor=THEME["accent_light"],
            padding=(6, 3),
        )
        style.map(
            "TButton",
            background=[
                ("active", THEME["accent_hover"]),
                ("disabled", THEME["accent_muted"]),
            ],
            foreground=[("disabled", "#e8f4ec")],
        )
        style.configure(
            "Accent.TButton",
            background=THEME["header"],
            bordercolor=THEME["header"],
            font=("TkDefaultFont", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", THEME["accent_hover"]),
                ("disabled", THEME["accent_muted"]),
            ],
        )
        style.configure(
            "TEntry",
            fieldbackground=THEME["surface"],
            foreground=THEME["text"],
            bordercolor=THEME["border"],
            lightcolor=THEME["border"],
            darkcolor=THEME["border"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=THEME["surface"],
            foreground=THEME["text"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["accent"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["surface"])],
            selectbackground=[("readonly", THEME["accent_light"])],
        )
        style.configure(
            "Treeview",
            background=THEME["surface"],
            fieldbackground=THEME["surface"],
            foreground=THEME["text"],
            bordercolor=THEME["border"],
        )
        style.configure(
            "Treeview.Heading",
            background=THEME["header"],
            foreground=THEME["text_on_dark"],
            relief="flat",
            bordercolor=THEME["header"],
        )
        style.map(
            "Treeview",
            background=[("selected", THEME["accent"])],
            foreground=[("selected", THEME["text_on_dark"])],
        )
        style.configure(
            "TNotebook",
            background=THEME["bg"],
            bordercolor=THEME["border"],
        )
        style.configure(
            "TNotebook.Tab",
            background=THEME["accent_light"],
            foreground=THEME["text"],
            padding=(8, 4),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", THEME["surface"])],
            expand=[("selected", (1, 1, 1, 0))],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=THEME["accent_light"],
            background=THEME["accent"],
            bordercolor=THEME["border"],
            lightcolor=THEME["accent"],
            darkcolor=THEME["accent"],
        )
        style.configure(
            "TSeparator",
            background=THEME["border"],
        )
        style.configure(
            "HeaderTitle.TLabel",
            background=THEME["header"],
            foreground=THEME["text_on_dark"],
            font=("TkDefaultFont", BASE_HEADER_TITLE_SIZE, "bold"),
        )
        style.configure(
            "HeaderSubtitle.TLabel",
            background=THEME["header"],
            foreground=THEME["text_subtle_on_dark"],
            font=("TkDefaultFont", BASE_HEADER_SUBTITLE_SIZE),
        )
        style.configure(
            "HeaderLogo.TLabel",
            background=THEME["header"],
        )
        style.configure(
            "TCheckbutton",
            background=THEME["bg"],
            foreground=THEME["text"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", THEME["accent_light"])],
        )
        style.configure(
            "Section.TCheckbutton",
            background=THEME["bg"],
            foreground=THEME["header"],
        )

    def _refresh_plot_toolbars(self) -> None:
        """Reserve enough vertical space for the matplotlib navigation buttons."""
        height = max(24, int(round(BASE_TOOLBAR_HEIGHT * self._ui_scale)))
        for toolbar in self._toolbars:
            toolbar._rescale()
            toolbar.configure(height=height)
            toolbar.pack_propagate(False)

    def _style_plot_axes(self, axis) -> None:
        """Give matplotlib axes the same soft green look as the rest of the UI."""
        axis.set_facecolor(THEME["surface"])
        for spine in axis.spines.values():
            spine.set_color(THEME["border"])
        axis.tick_params(colors=THEME["text_muted"])
        axis.xaxis.label.set_color(THEME["text"])
        axis.yaxis.label.set_color(THEME["text"])
        axis.title.set_color(THEME["header"])

    def _refresh_logo(self, scale: float) -> None:
        """Resize the header logo for the current display scale."""
        if not hasattr(self, "_logo_label"):
            return
        size = max(20, int(round(BASE_LOGO_SIZE * scale)))
        resized = self._logo_source.resize((size, size), Image.Resampling.LANCZOS)
        self._logo_photo = ImageTk.PhotoImage(resized)
        self._logo_label.configure(image=self._logo_photo)

    def _refresh_header_fonts(self, scale: float) -> None:
        """Keep the header typography in step with the UI scale."""
        if not hasattr(self, "_header_title"):
            return
        title_size = max(10, int(round(BASE_HEADER_TITLE_SIZE * scale)))
        subtitle_size = max(7, int(round(BASE_HEADER_SUBTITLE_SIZE * scale)))
        self._style.configure(
            "HeaderTitle.TLabel",
            font=("TkDefaultFont", title_size, "bold"),
        )
        self._style.configure(
            "HeaderSubtitle.TLabel",
            font=("TkDefaultFont", subtitle_size),
        )

    def apply_ui_scale(self, scale: float, *, remember: bool = True) -> None:
        """
        Resize fonts, table rows, and plots for a HiDPI display.

        Tk sizes almost everything from the named fonts, so rescaling those
        reflows the whole window; the table row height and the figure DPI are
        the two things that have to be nudged separately.
        """
        scale = clamp_ui_scale(scale)
        self._ui_scale = scale
        self.var_ui_scale.set(f"{scale:g}")

        self._set_tk_scaling(scale)
        for name, base in self._base_font_sizes.items():
            # A negative Tk font size means pixels rather than points; keep the
            # sign so the meaning survives scaling.
            sign = -1 if base < 0 else 1
            scaled = max(6, int(round(abs(base) * scale)))
            tkfont.nametofont(name).configure(size=sign * scaled)

        self._style.configure(
            "Treeview", rowheight=int(round(BASE_TREE_ROW_HEIGHT * scale))
        )
        for column, width in TREE_COLUMN_WIDTHS.items():
            self.tree.column(column, width=int(round(width * scale)))

        # Derive headings from the body font so both end up in the same units.
        # A positive Tk size means points, which `tk scaling` would multiply a
        # second time and leave the headings twice as large as everything else.
        body_size = tkfont.nametofont("TkDefaultFont").cget("size")
        sign = -1 if body_size < 0 else 1
        heading_size = sign * max(6, int(round(abs(body_size) * HEADING_FONT_RATIO)))
        # ttk.Checkbutton does not accept a -font option; style them instead.
        self._style.configure(
            "Section.TCheckbutton",
            font=("TkDefaultFont", heading_size, "bold"),
            foreground=THEME["header"],
        )
        self._filter_hint_label.configure(
            wraplength=int(round(BASE_HINT_WRAPLENGTH * scale))
        )
        self.progress.configure(length=int(round(BASE_PROGRESS_LENGTH * scale)))

        self._refresh_logo(scale)
        self._refresh_header_fonts(scale)

        for figure, canvas in (
            (self.figure, self.canvas_time),
            (self.figure_asd, self.canvas_asd),
        ):
            figure.tight_layout()
            canvas.draw_idle()
        self._refresh_plot_toolbars()

        self.minsize(*(int(round(value * scale)) for value in BASE_MIN_SIZE))
        self._resize_to_scale(scale)

        if remember:
            save_gui_state(self._gui_state())
            self.var_status.set(f"Display scale set to {scale:g}×")

    def _resize_to_scale(self, scale: float) -> None:
        """Grow the window with the scale, without spilling off the screen."""
        width = int(round(BASE_WINDOW_SIZE[0] * scale))
        height = int(round(BASE_WINDOW_SIZE[1] * scale))
        width = min(width, int(self.winfo_screenwidth() * 0.95))
        height = min(height, int(self.winfo_screenheight() * 0.92))
        self.geometry(f"{width}x{height}")

    def _on_scale_selected(self) -> None:
        try:
            scale = float(self.var_ui_scale.get())
        except ValueError:
            self.var_status.set(f"Display scale must be a number, got '{self.var_ui_scale.get()}'")
            return
        self.apply_ui_scale(scale)

    # ---------------------------------------------------------------- setup

    def _build_variables(self, data_root: str, dt: float, stored: GuiState) -> None:
        self.var_data_root = tk.StringVar(value=data_root)
        self.var_ui_scale = tk.StringVar(value=f"{self._ui_scale:g}")
        self.var_channels = tk.StringVar(value="XYZ")
        self.var_use_downsample = tk.BooleanVar(value=True)
        self.var_use_filter = tk.BooleanVar(value=True)
        self.var_use_postprocess = tk.BooleanVar(value=True)
        self.var_kaiser = tk.StringVar()
        self.var_highpass = tk.StringVar()
        self.var_lowpass = tk.StringVar()
        self.var_use_lowpass = tk.BooleanVar(value=True)
        self.var_order = tk.StringVar()
        self.var_zero_phase = tk.BooleanVar(value=True)
        self.var_use_trim = tk.BooleanVar(value=True)
        self.var_trim = tk.StringVar()
        self.var_use_window = tk.BooleanVar(value=True)
        self.var_use_segments = tk.BooleanVar(value=False)
        self.var_window = tk.StringVar(value="tukey")
        self.var_alpha = tk.StringVar()
        self.var_segment_days = tk.StringVar(value="")
        self.var_start = tk.StringVar(value="0")
        self.var_end = tk.StringVar(value="30")
        self.var_time_unit = tk.StringVar(value="days")
        self.var_status = tk.StringVar(value="Ready")
        self.var_memory = tk.StringVar(value="")
        self.var_cadence = tk.StringVar(value=str(dt))
        self.var_cadence_hint = tk.StringVar()
        self.var_filter_hint = tk.StringVar()
        self.var_plot_after_run = tk.BooleanVar(value=True)
        self.var_time_xmin = tk.StringVar()
        self.var_time_xmax = tk.StringVar()
        self.var_time_ymin = tk.StringVar()
        self.var_time_ymax = tk.StringVar()
        self.var_spec_xmin = tk.StringVar()
        self.var_spec_xmax = tk.StringVar()
        self.var_spec_ymin = tk.StringVar()
        self.var_spec_ymax = tk.StringVar()
        self.var_save_as_l1 = tk.BooleanVar(value=stored.save_as_l1)
        self.var_save_include_auxiliary = tk.BooleanVar(value=stored.save_include_auxiliary)
        self.var_skip_preprocessing = tk.BooleanVar(value=stored.skip_preprocessing)

        #: Last low-pass value the GUI derived from the cadence, so a hand-typed
        #: cutoff is never overwritten when the cadence changes.
        self._auto_lowpass: str | None = None
        self._downsample_widgets: list[tk.Widget] = []
        self._filter_widgets: list[tk.Widget] = []
        self._postprocess_widgets: list[tk.Widget] = []
        self._auxiliary_save_widgets: list[tk.Widget] = []

    def _gui_state(self) -> GuiState:
        return GuiState(
            data_root=self.var_data_root.get().strip(),
            ui_scale=self._ui_scale,
            save_as_l1=self.var_save_as_l1.get(),
            save_include_auxiliary=self.var_save_include_auxiliary.get(),
            skip_preprocessing=self.var_skip_preprocessing.get(),
        )

    def _sync_save_widgets(self) -> None:
        state = "normal" if self.var_save_as_l1.get() else "disabled"
        for widget in self._auxiliary_save_widgets:
            widget.configure(state=state)
        if not self.var_save_as_l1.get():
            self.var_save_include_auxiliary.set(False)

    def _bind_traces(self) -> None:
        """Wire up live hints once every widget and variable exists."""
        for variable in (
            self.var_channels,
            self.var_start,
            self.var_end,
            self.var_time_unit,
        ):
            variable.trace_add("write", lambda *_: self._update_memory_hint())
        self.var_cadence.trace_add("write", lambda *_: self._update_cadence_hint())
        self.var_lowpass.trace_add("write", lambda *_: self._update_filter_hint())
        self.var_use_lowpass.trace_add("write", lambda *_: self._update_filter_hint())
        for variable in (
            self.var_use_downsample,
            self.var_use_filter,
            self.var_use_postprocess,
        ):
            variable.trace_add("write", lambda *_: self._sync_stage_widgets())
        self._update_cadence_hint()

    def _build_layout(self) -> None:
        self._build_header()

        root_bar = ttk.Frame(self, padding=(8, 6, 8, 0))
        root_bar.pack(fill="x")
        ttk.Label(root_bar, text="Data directory").pack(side="left")
        entry = ttk.Entry(root_bar, textvariable=self.var_data_root)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _event: self.scan_data_root())
        ttk.Button(root_bar, text="Browse directory…", command=self.browse_data_root).pack(
            side="left"
        )
        ttk.Button(root_bar, text="Rescan", command=self.scan_data_root).pack(
            side="left", padx=(6, 0)
        )

        ttk.Separator(root_bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(root_bar, text="Display scale").pack(side="left")
        scale_box = ttk.Combobox(
            root_bar,
            textvariable=self.var_ui_scale,
            values=[f"{value:g}" for value in UI_SCALE_CHOICES],
            width=6,
        )
        scale_box.pack(side="left", padx=6)
        scale_box.bind("<<ComboboxSelected>>", lambda _event: self._on_scale_selected())
        scale_box.bind("<Return>", lambda _event: self._on_scale_selected())
        ttk.Button(root_bar, text="Apply", command=self._on_scale_selected).pack(side="left")

        # Vertical split so the package table and the plots can be resized
        # against each other instead of fighting for the same space.
        split = ttk.PanedWindow(self, orient="vertical")
        split.pack(fill="both", expand=True, padx=8, pady=6)

        top = ttk.Frame(split)
        bottom = ttk.Frame(split)
        split.add(top, weight=3)
        split.add(bottom, weight=2)

        panes = ttk.PanedWindow(top, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=1)

        self._build_package_panel(left)
        self._build_settings_panel(right)
        self._build_action_bar(bottom)
        self._build_save_options(bottom)
        self._build_plot_options(bottom)
        self._build_plot_area(bottom)

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", padding=(8, 5))
        header.pack(fill="x")

        self._logo_label = ttk.Label(header, style="HeaderLogo.TLabel")
        self._logo_label.pack(side="left")

        titles = ttk.Frame(header, style="Header.TFrame")
        titles.pack(side="left", padx=(10, 0))
        self._header_title = ttk.Label(
            titles,
            text="Mojito Barkeeper",
            style="HeaderTitle.TLabel",
        )
        self._header_title.pack(anchor="w")
        ttk.Label(
            titles,
            text="Mojito L1 data loading, combining, and preprocessing",
            style="HeaderSubtitle.TLabel",
        ).pack(anchor="w")

        self._refresh_logo(self._ui_scale)
        self._refresh_header_fonts(self._ui_scale)
        icon_size = max(16, int(round(24 * self._ui_scale)))
        icon = ImageTk.PhotoImage(
            self._logo_source.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        )
        self.iconphoto(True, icon)
        self._icon_photo = icon

    def _build_package_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Data packages  (click 'Use' to toggle, 'Op' to add/subtract)")
        box.pack(fill="both", expand=True)

        columns = ("use", "op", "package", "path")
        self.tree = ttk.Treeview(box, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("use", text="Use")
        self.tree.heading("op", text="Op")
        self.tree.heading("package", text="Package")
        self.tree.heading("path", text="Path")
        self.tree.column("use", width=44, anchor="center", stretch=False)
        self.tree.column("op", width=36, anchor="center", stretch=False)
        self.tree.column("package", width=300, anchor="w")
        self.tree.column("path", width=420, anchor="w")

        scroll = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))

        self.tree.tag_configure("enabled", background=THEME["accent_light"])
        self.tree.tag_configure("disabled", foreground=THEME["text_muted"])
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<Double-1>", lambda _event: self.toggle_enabled())

        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Add file…", command=self.add_file).pack(side="left")
        ttk.Button(buttons, text="Toggle use", command=self.toggle_enabled).pack(side="left", padx=4)
        ttk.Button(buttons, text="Toggle +/−", command=self.toggle_sign).pack(side="left")
        ttk.Button(buttons, text="Remove", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(buttons, text="Clear selection", command=self.disable_all).pack(side="left")

        window_box = ttk.LabelFrame(
            parent, text="Time selection — trims while reading (full files are ~5.6 GB each)"
        )
        window_box.pack(fill="x", pady=(8, 0))
        grid = ttk.Frame(window_box, padding=6)
        grid.pack(fill="x")
        ttk.Label(grid, text="Start time").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.var_start, width=10).grid(row=0, column=1, padx=(6, 16))
        ttk.Label(grid, text="End time").grid(row=0, column=2, sticky="w")
        ttk.Entry(grid, textvariable=self.var_end, width=10).grid(row=0, column=3, padx=6)
        ttk.Combobox(
            grid,
            textvariable=self.var_time_unit,
            values=list(TIME_UNITS),
            state="readonly",
            width=8,
        ).grid(row=0, column=4, padx=(0, 8))
        ttk.Label(grid, text="from the file start (blank end = to the end)").grid(
            row=0, column=5, sticky="w"
        )
        ttk.Label(grid, textvariable=self.var_memory, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )

    def _track_widget(self, group: str, widget: tk.Widget) -> tk.Widget:
        """Register a parameter widget so it can be greyed out with its stage."""
        getattr(self, f"_{group}_widgets").append(widget)
        return widget

    def _sync_stage_widgets(self) -> None:
        """Enable parameter fields only for pipeline stages that are turned on."""
        if self.var_skip_preprocessing.get():
            for group in ("downsample", "filter", "postprocess"):
                for widget in getattr(self, f"_{group}_widgets"):
                    try:
                        widget.configure(state="disabled")
                    except tk.TclError:
                        pass
            for toggle in self._section_toggles:
                toggle.configure(state="disabled")
            return

        for toggle in self._section_toggles:
            toggle.configure(state="normal")
        stages = {
            "downsample": self.var_use_downsample.get(),
            "filter": self.var_use_filter.get(),
            "postprocess": self.var_use_postprocess.get(),
        }
        for group, enabled in stages.items():
            state = "normal" if enabled else "disabled"
            for widget in getattr(self, f"_{group}_widgets"):
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _section_toggle(
        self, parent: ttk.Frame, text: str, variable: tk.BooleanVar
    ) -> ttk.Checkbutton:
        """Section heading that also enables or disables that pipeline stage."""
        toggle = ttk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=self._sync_stage_widgets,
            style="Section.TCheckbutton",
        )
        self._section_toggles.append(toggle)
        return toggle

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="process_pipeline parameters")
        box.pack(fill="both", expand=True)
        grid = ttk.Frame(box, padding=8)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(grid, text="Channels").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            grid,
            textvariable=self.var_channels,
            values=list(CHANNEL_SETS),
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Separator(grid, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )

        row += 1
        ttk.Checkbutton(
            grid,
            text="Skip preprocessing (load/combine only)",
            variable=self.var_skip_preprocessing,
            command=self._sync_stage_widgets,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(
            grid,
            text="Use for saved pipeline outputs to avoid downsampling/trim again",
            style="Muted.TLabel",
        ).grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Separator(grid, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )

        row += 1
        self._section_toggle(grid, "Downsample", self.var_use_downsample).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(grid, text="Target cadence [s]").grid(row=row, column=0, sticky="w", pady=3)
        cadence = self._track_widget(
            "downsample",
            ttk.Combobox(
                grid,
                textvariable=self.var_cadence,
                values=list(CADENCE_LABELS),
                width=18,
            ),
        )
        cadence.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(grid, textvariable=self.var_cadence_hint, style="Muted.TLabel").grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        ttk.Label(grid, text="Kaiser beta").grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "downsample",
            ttk.Entry(grid, textvariable=self.var_kaiser, width=14),
        ).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Separator(grid, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )

        row += 1
        self._section_toggle(grid, "Filter", self.var_use_filter).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(grid, text="Highpass [Hz]").grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "filter",
            ttk.Entry(grid, textvariable=self.var_highpass, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Checkbutton(grid, text="Lowpass [Hz]", variable=self.var_use_lowpass).grid(
            row=row, column=0, sticky="w", pady=3
        )
        self._track_widget(
            "filter",
            ttk.Entry(grid, textvariable=self.var_lowpass, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        self._filter_hint_label = ttk.Label(
            grid,
            textvariable=self.var_filter_hint,
            style="Error.TLabel",
            wraplength=BASE_HINT_WRAPLENGTH,
        )
        self._filter_hint_label.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(grid, text="Order").grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "filter",
            ttk.Entry(grid, textvariable=self.var_order, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        self._track_widget(
            "filter",
            ttk.Checkbutton(grid, text="Zero phase", variable=self.var_zero_phase),
        ).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Separator(grid, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )

        row += 1
        self._section_toggle(
            grid, "Trim / window / segments", self.var_use_postprocess
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self._track_widget(
            "postprocess",
            ttk.Checkbutton(grid, text="Trim fraction", variable=self.var_use_trim),
        ).grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "postprocess",
            ttk.Entry(grid, textvariable=self.var_trim, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        self._track_widget(
            "postprocess",
            ttk.Checkbutton(grid, text="Window", variable=self.var_use_window),
        ).grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "postprocess",
            ttk.Combobox(
                grid,
                textvariable=self.var_window,
                values=list(WINDOW_TYPES),
                state="readonly",
                width=12,
            ),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(grid, text="Window alpha").grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "postprocess",
            ttk.Entry(grid, textvariable=self.var_alpha, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        self._track_widget(
            "postprocess",
            ttk.Checkbutton(grid, text="Segment days", variable=self.var_use_segments),
        ).grid(row=row, column=0, sticky="w", pady=3)
        self._track_widget(
            "postprocess",
            ttk.Entry(grid, textvariable=self.var_segment_days, width=14),
        ).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(grid, text="(blank segment length = one segment)", style="Muted.TLabel").grid(
            row=row, column=1, sticky="w"
        )

        row += 1
        ttk.Button(grid, text="Reset to search-config defaults", command=self._reset_settings).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self._sync_stage_widgets()

    def _build_action_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent, padding=(0, 4, 0, 6))
        bar.pack(fill="x")
        self.run_button = ttk.Button(
            bar, text="Run pipeline", command=self.run_pipeline, style="Accent.TButton"
        )
        self.run_button.pack(side="left")
        self.save_button = ttk.Button(
            bar, text="Save result…", command=self.save_result, state="disabled"
        )
        self.save_button.pack(side="left", padx=6)
        self.plot_button = ttk.Button(
            bar, text="Plot", command=self.plot_result, state="disabled"
        )
        self.plot_button.pack(side="left")
        self.progress = ttk.Progressbar(
            bar, mode="indeterminate", length=BASE_PROGRESS_LENGTH
        )
        self.progress.pack(side="left", padx=10)
        ttk.Label(bar, textvariable=self.var_status).pack(side="left", padx=6)

    def _build_save_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Save options")
        box.pack(fill="x", pady=(0, 6))
        row = ttk.Frame(box, padding=(6, 4))
        row.pack(fill="x")
        ttk.Checkbutton(
            row,
            text="Save as Mojito L1 (reloadable in this pipeline)",
            variable=self.var_save_as_l1,
            command=self._sync_save_widgets,
        ).pack(side="left")
        auxiliary = ttk.Checkbutton(
            row,
            text="Include auxiliary data from first source (LTTs, orbits, noise)",
            variable=self.var_save_include_auxiliary,
        )
        auxiliary.pack(side="left", padx=(12, 0))
        self._auxiliary_save_widgets = [auxiliary]
        self._sync_save_widgets()

    def _build_plot_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Plot options")
        box.pack(fill="x", pady=(0, 4))

        top = ttk.Frame(box, padding=(4, 2, 4, 0))
        top.pack(fill="x")
        ttk.Checkbutton(
            top,
            text="Plot after run",
            variable=self.var_plot_after_run,
        ).pack(side="left")
        ttk.Label(
            top,
            text="(leave axis limits blank for autoscale)",
            style="Muted.TLabel",
        ).pack(side="left", padx=(12, 0))

        grid = ttk.Frame(box, padding=4)
        grid.pack(fill="x")

        def axis_row(row: int, label: str, xmin, xmax, ymin, ymax) -> None:
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(grid, text="x min").grid(row=row, column=1, sticky="e", padx=(8, 2))
            ttk.Entry(grid, textvariable=xmin, width=10).grid(row=row, column=2, sticky="w")
            ttk.Label(grid, text="x max").grid(row=row, column=3, sticky="e", padx=(8, 2))
            ttk.Entry(grid, textvariable=xmax, width=10).grid(row=row, column=4, sticky="w")
            ttk.Label(grid, text="y min").grid(row=row, column=5, sticky="e", padx=(8, 2))
            ttk.Entry(grid, textvariable=ymin, width=10).grid(row=row, column=6, sticky="w")
            ttk.Label(grid, text="y max").grid(row=row, column=7, sticky="e", padx=(8, 2))
            ttk.Entry(grid, textvariable=ymax, width=10).grid(row=row, column=8, sticky="w")

        axis_row(
            0,
            "Time series [d, TDI]",
            self.var_time_xmin,
            self.var_time_xmax,
            self.var_time_ymin,
            self.var_time_ymax,
        )
        axis_row(
            1,
            "Spectrum [Hz, ASD]",
            self.var_spec_xmin,
            self.var_spec_xmax,
            self.var_spec_ymin,
            self.var_spec_ymax,
        )

    def _build_plot_area(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        # Small requested size; the canvas expands to fill whatever the pane gives it.
        self.figure = Figure(figsize=(7, 3), dpi=BASE_FIGURE_DPI, facecolor=THEME["bg"])
        self.ax_time = self.figure.add_subplot(111)
        self.ax_time.set_xlabel("time [days]")
        self.ax_time.set_ylabel("TDI")
        self.ax_time.grid(alpha=0.45, color=THEME["plot_grid"])
        self._style_plot_axes(self.ax_time)

        self.figure_asd = Figure(figsize=(7, 3), dpi=BASE_FIGURE_DPI, facecolor=THEME["bg"])
        self.ax_asd = self.figure_asd.add_subplot(111)
        self.ax_asd.set_xlabel("frequency [Hz]")
        self.ax_asd.set_ylabel("ASD [1/sqrt(Hz)]")
        self._style_plot_axes(self.ax_asd)

        for figure, title in ((self.figure, "Time series"), (self.figure_asd, "Spectrum")):
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            frame.rowconfigure(1, weight=0)

            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=1, column=0, sticky="ew")
            self._toolbars.append(toolbar)
            if figure is self.figure:
                self.canvas_time = canvas
            else:
                self.canvas_asd = canvas

        self._refresh_plot_toolbars()

        self.figure.tight_layout()
        self.figure_asd.tight_layout()

    # ------------------------------------------------------------ packages

    def browse_data_root(self) -> None:
        chosen = filedialog.askdirectory(
            title="Choose the directory holding the Mojito L1 packages",
            initialdir=self.var_data_root.get() or os.getcwd(),
        )
        if chosen:
            self.var_data_root.set(chosen)
            self.scan_data_root()

    def scan_data_root(self) -> None:
        root = self.var_data_root.get().strip()
        if not root:
            self.var_status.set("Choose a data directory with 'Browse…' to list packages")
            return
        if not os.path.isdir(root):
            self.var_status.set(f"Not a directory: {root}")
            return

        found = discover_packages(root)
        self.packages = found
        self._refresh_tree()
        if found:
            save_gui_state(self._gui_state())
            self.var_status.set(f"Found {len(found)} package(s) under {root}")
        else:
            self.var_status.set(f"No .h5 or .hdf5 files found under {root}")

    def add_file(self) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=self.var_data_root.get() or os.getcwd(),
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not chosen:
            return
        if any(package.path == chosen for package in self.packages):
            self.var_status.set("That file is already listed")
            return
        self.packages.append(DataPackage(path=chosen, enabled=True))
        self._refresh_tree()

    def _selected_index(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def _on_tree_click(self, event: tk.Event) -> None:
        if self.tree.identify("region", event.x, event.y) != "cell":
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        column = self.tree.identify_column(event.x)
        if column == "#1":
            self._toggle_index(int(row), what="enabled")
        elif column == "#2":
            self._toggle_index(int(row), what="sign")

    def _toggle_index(self, index: int, *, what: str) -> None:
        package = self.packages[index]
        if what == "enabled":
            package.enabled = not package.enabled
            if package.enabled and is_saved_pipeline_output(package.path):
                self.var_skip_preprocessing.set(True)
                self._sync_stage_widgets()
                self.var_status.set(
                    f"Enabled saved pipeline output — preprocessing skipped for {package.label}"
                )
        else:
            package.sign = -package.sign
            package.enabled = True
        self._refresh_tree(keep_selection=index)

    def toggle_enabled(self) -> None:
        index = self._selected_index()
        if index is not None:
            self._toggle_index(index, what="enabled")

    def toggle_sign(self) -> None:
        index = self._selected_index()
        if index is not None:
            self._toggle_index(index, what="sign")

    def remove_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        del self.packages[index]
        self._refresh_tree()

    def disable_all(self) -> None:
        for package in self.packages:
            package.enabled = False
        self._refresh_tree()

    def _refresh_tree(self, keep_selection: int | None = None) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, package in enumerate(self.packages):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    "✓" if package.enabled else "",
                    package.operation if package.enabled else "",
                    package.label,
                    package.path,
                ),
                tags=("enabled",) if package.enabled else ("disabled",),
            )
        if keep_selection is not None and 0 <= keep_selection < len(self.packages):
            self.tree.selection_set(str(keep_selection))
        self._update_memory_hint()

    def _update_memory_hint(self) -> None:
        enabled = [package for package in self.packages if package.enabled]
        channels = CHANNEL_SETS.get(self.var_channels.get(), XYZ_CHANNELS)
        try:
            window = self._collect_window()
        except PipelineError as error:
            self.var_memory.set(str(error))
            return

        # Without an end time the whole 731-day file is read.
        seconds = window.duration if window.duration is not None else 731 * 86400.0
        samples = seconds * NOMINAL_SOURCE_FS
        total_bytes = samples * len(channels) * BYTES_PER_SAMPLE * max(len(enabled), 1)
        self.var_memory.set(
            f"{len(enabled)} package(s) × {len(channels)} channel(s) × "
            f"{seconds / 86400:g} day(s) ≈ {total_bytes / 1e9:.2f} GB peak in memory"
        )

    def _update_cadence_hint(self) -> None:
        try:
            cadence = self._parse_cadence()
        except PipelineError as error:
            self.var_cadence_hint.set(str(error))
            return

        if cadence is None:
            self.var_cadence_hint.set("keeps the native rate; the low-pass still applies")
        else:
            self.var_cadence_hint.set(
                f"= {1.0 / cadence:g} Hz, Nyquist {0.5 / cadence:g} Hz"
            )
            # Track the cadence with the anti-alias cutoff the way
            # mojito_preprocessing_pipeline_kwargs does, unless it was hand-edited.
            nyquist = f"{0.5 / cadence:g}"
            if self.var_lowpass.get().strip() == (self._auto_lowpass or ""):
                self.var_lowpass.set(nyquist)
            self._auto_lowpass = nyquist

        self._update_filter_hint()

    def _update_filter_hint(self) -> None:
        """Warn when a hand-set low-pass would alias at the chosen cadence."""
        self.var_filter_hint.set("")
        if not self.var_use_filter.get() or not self.var_use_lowpass.get():
            return
        try:
            cadence = self._parse_cadence()
            lowpass = float(self.var_lowpass.get())
        except (PipelineError, ValueError):
            return
        # The box holds the Nyquist rounded to six significant figures, so the
        # tolerance has to exceed that rounding error or it warns about itself.
        if cadence is not None and lowpass > (0.5 / cadence) * (1 + 1e-4):
            self.var_filter_hint.set(
                f"above the {0.5 / cadence:g} Hz Nyquist of the target cadence — "
                "content there will alias"
            )

    # ------------------------------------------------------------ settings

    def _parse_cadence(self) -> float | None:
        """Read the cadence box; ``None`` means keep the native sampling."""
        text = self.var_cadence.get().strip()
        if not text or text == NATIVE_CADENCE:
            return None
        try:
            cadence = float(text)
        except ValueError as error:
            raise PipelineError(
                f"Target cadence must be a number of seconds or '{NATIVE_CADENCE}', "
                f"got '{text}'"
            ) from error
        if cadence <= 0:
            raise PipelineError("Target cadence must be positive")
        return cadence

    def _collect_window(self) -> TimeWindow:
        """Read the start/end boxes into a :class:`TimeWindow`."""
        unit = self.var_time_unit.get()
        start_text = self.var_start.get().strip()
        end_text = self.var_end.get().strip()
        try:
            start = float(start_text) if start_text else 0.0
            end = float(end_text) if end_text else None
        except ValueError as error:
            raise PipelineError(
                f"Start and end time must be numbers in {unit}, "
                f"got '{start_text}' and '{end_text}'"
            ) from error

        try:
            return TimeWindow.from_unit(start, end, unit)
        except ValueError as error:
            raise PipelineError(str(error)) from error

    def _reset_settings(self) -> None:
        try:
            cadence = self._parse_cadence()
        except PipelineError:
            cadence = 10.0
        settings = PipelineSettings.from_search_config(
            dt=cadence, channels=self.var_channels.get()
        )
        self.var_kaiser.set(f"{settings.kaiser_beta:g}")
        self.var_highpass.set(f"{settings.highpass_cutoff:g}")
        self._auto_lowpass = f"{settings.lowpass_cutoff:g}"
        self.var_lowpass.set(self._auto_lowpass)
        self.var_use_lowpass.set(True)
        self.var_order.set(str(settings.filter_order))
        self.var_zero_phase.set(settings.zero_phase)
        self.var_use_downsample.set(True)
        self.var_use_filter.set(True)
        self.var_use_postprocess.set(True)
        self.var_use_trim.set(True)
        self.var_trim.set(f"{settings.trim_fraction:g}")
        self.var_use_window.set(True)
        self.var_use_segments.set(False)
        self.var_window.set(settings.window)
        self.var_alpha.set(f"{settings.window_alpha:g}")
        self.var_segment_days.set("")
        self._sync_stage_widgets()
        self.var_status.set("Parameters reset to GB_search_config defaults")

    def _collect_settings(self) -> PipelineSettings:
        def as_float(variable: tk.StringVar, name: str) -> float:
            try:
                return float(variable.get())
            except ValueError as error:
                raise PipelineError(f"{name} must be a number, got '{variable.get()}'") from error

        def as_int(variable: tk.StringVar, name: str) -> int:
            try:
                return int(float(variable.get()))
            except ValueError as error:
                raise PipelineError(f"{name} must be an integer, got '{variable.get()}'") from error

        segment_text = self.var_segment_days.get().strip()
        use_postprocess = self.var_use_postprocess.get()
        use_segments = use_postprocess and self.var_use_segments.get()
        truncate_days = None
        if use_segments and segment_text:
            truncate_days = as_float(self.var_segment_days, "Segment days")

        return PipelineSettings(
            channels=CHANNEL_SETS[self.var_channels.get()],
            target_dt=self._parse_cadence(),
            kaiser_beta=as_float(self.var_kaiser, "Kaiser beta"),
            highpass_cutoff=as_float(self.var_highpass, "Highpass cutoff"),
            lowpass_cutoff=as_float(self.var_lowpass, "Lowpass cutoff")
            if self.var_use_lowpass.get()
            else None,
            filter_order=as_int(self.var_order, "Filter order"),
            zero_phase=self.var_zero_phase.get(),
            trim_fraction=as_float(self.var_trim, "Trim fraction"),
            truncate_days=truncate_days,
            window=self.var_window.get(),
            window_alpha=as_float(self.var_alpha, "Window alpha"),
            apply_preprocessing=not self.var_skip_preprocessing.get(),
            apply_downsample=self.var_use_downsample.get(),
            apply_filter=self.var_use_filter.get(),
            apply_window=use_postprocess and self.var_use_window.get(),
            apply_trim=use_postprocess and self.var_use_trim.get(),
            apply_segments=use_segments,
        )

    # ----------------------------------------------------------------- run

    def run_pipeline(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        try:
            settings = self._collect_settings()
            window = self._collect_window()
        except (PipelineError, ValueError) as error:
            messagebox.showerror("Invalid parameters", str(error))
            return

        enabled = [package for package in self.packages if package.enabled]
        if not enabled:
            messagebox.showwarning(
                "No packages selected",
                "Enable at least one data package by clicking its 'Use' cell.",
            )
            return

        selection = [DataPackage(p.path, p.label, p.sign, p.enabled) for p in enabled]
        self.run_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.plot_button.configure(state="disabled")
        self.progress.start(12)
        self.var_status.set("Starting…")

        def work() -> None:
            try:
                result = process(
                    selection,
                    settings,
                    window=window,
                    progress=lambda message: self._messages.put(("status", message)),
                )
                self._messages.put(("done", result))
            except Exception as error:  # surfaced in the UI, not swallowed
                self._messages.put(("error", (error, traceback.format_exc())))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self._messages.get_nowait()
                if kind == "status":
                    self.var_status.set(payload)
                elif kind == "done":
                    self._on_success(payload)
                elif kind == "error":
                    self._on_failure(*payload)
        except queue.Empty:
            pass
        self._drain_job = self.after(100, self._drain_messages)

    def destroy(self) -> None:
        """Stop the polling callback before the widgets go away."""
        if self._drain_job is not None:
            self.after_cancel(self._drain_job)
            self._drain_job = None
        super().destroy()

    def _on_success(self, result) -> None:
        self.result = result
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.plot_button.configure(state="normal")
        self.var_status.set(
            f"Done: {result.summary} → "
            f"{len(result.series[result.channels[0]])} samples at dt={result.dt:g} s "
            f"({result.duration / 86400:.2f} days, {result.n_segments} segment(s))"
        )
        if self.var_plot_after_run.get():
            try:
                self._plot(result)
            except PipelineError as error:
                messagebox.showerror("Plot failed", str(error))

    def plot_result(self) -> None:
        """Redraw the last processed result (e.g. after changing axis limits)."""
        if self.result is None:
            return
        try:
            self._plot(self.result)
        except PipelineError as error:
            messagebox.showerror("Plot failed", str(error))

    def _on_failure(self, error: Exception, formatted: str) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.var_status.set(f"Failed: {error}")
        print(formatted)
        messagebox.showerror("Pipeline failed", f"{type(error).__name__}: {error}")

    # --------------------------------------------------------------- plots

    def _optional_limit(self, variable: tk.StringVar, name: str) -> float | None:
        text = variable.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as error:
            raise PipelineError(f"{name} must be a number or blank, got '{text}'") from error

    def _axis_limits(
        self,
        xmin_var: tk.StringVar,
        xmax_var: tk.StringVar,
        ymin_var: tk.StringVar,
        ymax_var: tk.StringVar,
        *,
        prefix: str,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        xmin = self._optional_limit(xmin_var, f"{prefix} x min")
        xmax = self._optional_limit(xmax_var, f"{prefix} x max")
        ymin = self._optional_limit(ymin_var, f"{prefix} y min")
        ymax = self._optional_limit(ymax_var, f"{prefix} y max")
        if xmin is not None and xmax is not None and xmin >= xmax:
            raise PipelineError(f"{prefix} x min must be less than x max")
        if ymin is not None and ymax is not None and ymin >= ymax:
            raise PipelineError(f"{prefix} y min must be less than y max")
        if prefix == "Spectrum":
            for value, name in (
                (xmin, "x min"),
                (xmax, "x max"),
                (ymin, "y min"),
                (ymax, "y max"),
            ):
                if value is not None and value <= 0:
                    raise PipelineError(f"Spectrum {name} must be positive on a log scale")
        return xmin, xmax, ymin, ymax

    @staticmethod
    def _apply_axis_limits(
        axis,
        xmin: float | None,
        xmax: float | None,
        ymin: float | None,
        ymax: float | None,
    ) -> None:
        if xmin is not None or xmax is not None:
            axis.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            axis.set_ylim(bottom=ymin, top=ymax)

    def _plot(self, result) -> None:
        """Draw the first channel: breakdown plus total, or a lone signal on its own."""
        title = f"{result.summary}   ({result.duration / 86400:.2f} d @ dt={result.dt:g} s)"
        channel = result.channels[0]
        time_limits = self._axis_limits(
            self.var_time_xmin,
            self.var_time_xmax,
            self.var_time_ymin,
            self.var_time_ymax,
            prefix="Time series",
        )
        spec_limits = self._axis_limits(
            self.var_spec_xmin,
            self.var_spec_xmax,
            self.var_spec_ymin,
            self.var_spec_ymax,
            prefix="Spectrum",
        )

        components = result.components
        single_signal = len(components) == 1 or (
            not components and len(result.contributions) == 1
        )

        self.ax_time.clear()
        times = (result.times()  / 86400.0 - 1131.133)
        if single_signal and components:
            component = components[0]
            index, values = decimate_for_display(component.series[channel])
            self.ax_time.plot(
                times[index],
                values,
                lw=0.9,
                color=COMPONENT_COLORS[0],
                alpha=0.9,
                label=component.signed_label,
            )
        elif single_signal:
            label = result.contributions[0]
            index, values = decimate_for_display(result.series[channel])
            self.ax_time.plot(
                times[index],
                values,
                lw=0.9,
                color=COMPONENT_COLORS[0],
                alpha=0.9,
                label=label,
            )
        else:
            for component, color in zip(components, COMPONENT_COLORS):
                index, values = decimate_for_display(component.series[channel])
                self.ax_time.plot(
                    times[index],
                    values,
                    lw=0.7,
                    color=color,
                    alpha=0.8,
                    label=component.signed_label,
                )
            index, values = decimate_for_display(result.series[channel])
            self.ax_time.plot(
                times[index],
                values,
                lw=0.9,
                color=THEME["plot_total"],
                label="total",
                zorder=0,
            )
        self.ax_time.set_xlabel("time [days]")
        self.ax_time.set_ylabel(f"TDI {channel}")
        self.ax_time.set_title(title, fontsize=9)
        self.ax_time.legend(loc="upper right", fontsize=8, framealpha=0.92)
        self.ax_time.grid(alpha=0.45, color=THEME["plot_grid"])
        self._style_plot_axes(self.ax_time)
        self._apply_axis_limits(self.ax_time, *time_limits)
        self.figure.tight_layout()
        self.canvas_time.draw_idle()

        self.ax_asd.clear()
        if single_signal and components:
            component = components[0]
            freqs, asd = amplitude_spectral_density(component.series[channel], result.dt)
            self.ax_asd.loglog(
                freqs,
                asd,
                lw=0.9,
                color=COMPONENT_COLORS[0],
                alpha=0.9,
                label=component.signed_label,
            )
        elif single_signal:
            label = result.contributions[0]
            freqs, asd = result.spectrum(channel)
            self.ax_asd.loglog(
                freqs,
                asd,
                lw=0.9,
                color=COMPONENT_COLORS[0],
                alpha=0.9,
                label=label,
            )
        else:
            for component, color in zip(components, COMPONENT_COLORS):
                freqs, asd = amplitude_spectral_density(component.series[channel], result.dt)
                self.ax_asd.loglog(
                    freqs,
                    asd,
                    lw=0.7,
                    color=color,
                    alpha=0.8,
                    label=component.signed_label,
                )
            freqs, asd = result.spectrum(channel)
            self.ax_asd.loglog(
                freqs,
                asd,
                lw=0.9,
                color=THEME["plot_total"],
                label="total",
                zorder=0,
            )
        self.ax_asd.set_xlabel("frequency [Hz]")
        self.ax_asd.set_ylabel(f"ASD of TDI {channel} [1/sqrt(Hz)]")
        self.ax_asd.set_title(title, fontsize=9)
        self.ax_asd.legend(loc="lower left", fontsize=8, framealpha=0.92)
        self.ax_asd.grid(alpha=0.45, color=THEME["plot_grid"], which="both")
        self._style_plot_axes(self.ax_asd)
        self._apply_axis_limits(self.ax_asd, *spec_limits)
        self.figure_asd.tight_layout()


        self.canvas_asd.draw_idle()

    def save_result(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5")],
            initialfile="mojito_processed.h5",
        )
        if not path:
            return
        try:
            save_result(
                path,
                self.result,
                options=SaveOptions(
                    as_l1=self.var_save_as_l1.get(),
                    include_auxiliary=self.var_save_include_auxiliary.get(),
                ),
            )
            save_gui_state(self._gui_state())
        except Exception as error:
            messagebox.showerror("Save failed", f"{type(error).__name__}: {error}")
            return
        self.var_status.set(f"Saved {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Directory scanned for Mojito L1 packages. Defaults to the directory "
            "used last time; otherwise choose one with 'Browse directory…'."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=10.0,
        help="Initial target cadence in seconds",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=None,
        help=(
            "Display scale for fonts, tables, and plots (e.g. 2 on a 4K screen). "
            "Defaults to the remembered value, else a guess from the screen size."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    app = MojitoPipelineGUI(data_root=args.data_root, dt=args.dt, ui_scale=args.ui_scale)
    app.mainloop()


if __name__ == "__main__":
    main()
