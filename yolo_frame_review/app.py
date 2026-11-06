"""Tkinter GUI: browse long videos, compare with YOLO twin, save frames."""
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import session, yolo_pairing
from .video import VIDEO_EXTS, ScrubWorker, position_cap, read_frame


class FrameReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Frame Review")

        # State variables
        self.video_folder = None
        self.videos = []
        self.current_video = None
        self.current_video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame_idx = 0

        # Optional YOLO-annotated twin of the current video — display only.
        # Save/delete/smart-skip always use self.cap (original pixels).
        self.yolo_cap = None
        self.yolo_path = None
        # rel video path -> YOLO video path, filled by "Attach YOLO folder";
        # load_video auto-attaches from it
        self.yolo_map = {}

        # Slider drag state
        self.slider_dragging = False
        self._slider_after_id = None
        self.slider_items = {}

        # Scrub worker — decodes drag-preview frames off the main thread.
        # _scrub_gen is bumped whenever the shown video changes so stale
        # in-flight results get dropped.
        self._scrub_gen = 0
        self._scrub = ScrubWorker(
            lambda f, i, g: self.root.after(0, lambda: self._on_scrub_result(f, i, g)))

        # Canvas image item — reused instead of delete+recreate every frame
        self.canvas_img_id = None

        # Last decoded full-res frame — lets save/export reuse pixels
        # already on screen instead of re-seeking + re-decoding
        self._cached_idx = -1
        self._cached_rgb = None

        # Last frame put on screen (pre-zoom, original or YOLO) — zoom/pan
        # re-render from it instead of re-seeking the decoder
        self._shown_rgb = None

        # Zoom state
        self.zoom_scale = 1.0
        self.zoom_center = (0.5, 0.5)  # normalized (cx, cy)

        # Playback state
        self.is_playing = False

        # Guard flag: suppress <<ListboxSelect>> during programmatic list rebuilds
        self._updating_list = False

        # Session tracking
        self.session_file = session.SESSION_FILE
        self.frames_data = {"files": []}

        today = datetime.now().strftime("%Y-%m-%d")
        self.output_folder = f"extracted_frames_{today}"

        self.load_session()
        # video_notes must init after load_session so frames_data is populated
        self.video_notes = self.frames_data.get("notes", {})
        self.setup_gui()
        # Open at the layout's natural size, clamped to screen (minus
        # taskbar/title bar) — smaller laptops fall back to scrollbars
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        rw, rh = self._page.winfo_reqwidth(), self._page.winfo_reqheight()
        h = min(rh, sh - 90)
        # Height clamped → vertical bar will show; widen by its width so it
        # doesn't also force a horizontal bar
        extra_w = self._page_vbar.winfo_reqwidth() if h < rh else 0
        w = min(rw + extra_w, sw - 20)
        self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+0")
        self.setup_keyboard_shortcuts()
        # reflect any prior session data in F/V label
        total_f, total_v = self._get_session_totals()
        self.fv_label.config(text=f"F/V: {total_f}/{total_v}")

    # ──────────────────────────────────────────────────────────────────
    # GUI setup
    # ──────────────────────────────────────────────────────────────────

    def setup_gui(self):
        BG      = "#111111"
        BG_MID  = "#181818"
        BG_BAR  = "#1d2b38"
        BG_BTN  = "#252525"
        BG_BTN2 = "#2e2e2e"
        BG_CTRL = "#1e1e1e"
        BG_SLD  = "#161616"
        FG      = "#f0f0f0"
        FG_DIM  = "#9a9a9a"
        FG_PATH = "#7aaccc"
        AC_BLUE  = "#4a9eda"
        AC_GREEN = "#3dba72"
        AC_RED   = "#d9534f"
        AC_ORA   = "#d4863a"

        self.root.configure(bg=BG)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.Vertical.TScrollbar",
                        background="#2a2a2a", troughcolor=BG,
                        bordercolor=BG, arrowcolor="#4a4a4a",
                        darkcolor="#2a2a2a", lightcolor="#2a2a2a",
                        relief="flat")
        style.map("Dark.Vertical.TScrollbar",
                  background=[("active", "#3a3a3a"), ("pressed", "#4a4a4a")])
        style.configure("Dark.Horizontal.TScrollbar",
                        background="#2a2a2a", troughcolor=BG,
                        bordercolor=BG, arrowcolor="#4a4a4a",
                        darkcolor="#2a2a2a", lightcolor="#2a2a2a",
                        relief="flat")
        style.map("Dark.Horizontal.TScrollbar",
                  background=[("active", "#3a3a3a"), ("pressed", "#4a4a4a")])

        # Scrollable page — whole UI lives inside a canvas so it stays reachable
        # on screens smaller than the layout; bars show only on overflow
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        self._page_outer = outer
        self.page_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        self._page_vbar = ttk.Scrollbar(outer, orient=tk.VERTICAL,
                                        command=self.page_canvas.yview,
                                        style="Dark.Vertical.TScrollbar")
        self._page_hbar = ttk.Scrollbar(outer, orient=tk.HORIZONTAL,
                                        command=self.page_canvas.xview,
                                        style="Dark.Horizontal.TScrollbar")
        self.page_canvas.configure(yscrollcommand=self._page_vbar.set,
                                   xscrollcommand=self._page_hbar.set)
        self.page_canvas.grid(row=0, column=0, sticky="nsew")

        page = tk.Frame(self.page_canvas, bg=BG)
        self._page = page
        self._page_win = self.page_canvas.create_window(0, 0, window=page, anchor="nw")
        page.bind("<Configure>", self._fit_page)
        outer.bind("<Configure>", self._fit_page)
        self.root.bind_all("<MouseWheel>", self._on_page_wheel, add="+")
        self.root.bind_all("<Shift-MouseWheel>", self._on_page_wheel, add="+")

        # Top bar
        top_frame = tk.Frame(page, bg=BG_BAR, pady=10)
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text="Video Folder:", font=("Arial", 10, "bold"),
                 bg=BG_BAR, fg=FG).pack(side=tk.LEFT, padx=10)
        self.folder_label = tk.Label(top_frame, text="No folder selected",
                                     font=("Arial", 10), bg=BG_BAR, fg=FG_PATH)
        self.folder_label.pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Browse Folder", command=self.browse_folder,
                  bg=AC_GREEN, fg="white", font=("Arial", 10, "bold"),
                  padx=15, cursor="hand2", relief=tk.FLAT).pack(side=tk.LEFT, padx=10)
        # Disabled until Browse Folder has loaded videos (enabled in load_videos)
        self.yolo_folder_btn = tk.Button(
            top_frame, text="Attach YOLO Folder", command=self.attach_yolo_folder,
            bg=AC_ORA, fg="white", font=("Arial", 10, "bold"),
            disabledforeground="#6a6a6a", state=tk.DISABLED,
            padx=15, cursor="hand2", relief=tk.FLAT)
        self.yolo_folder_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.output_folder_label = tk.Label(top_frame,
                                            text=Path(self.output_folder).name or self.output_folder,
                                            font=("Arial", 10), bg=BG_BAR, fg=FG_PATH)
        self.output_folder_label.pack(side=tk.RIGHT, padx=(0, 10))
        tk.Label(top_frame, text="Output Folder:", font=("Arial", 10, "bold"),
                 bg=BG_BAR, fg=FG).pack(side=tk.RIGHT, padx=(10, 0))

        self.fv_label = tk.Label(top_frame, text="F/V: 0/0",
                                 font=("Consolas", 11, "bold"), bg=BG_BAR, fg=AC_GREEN)
        self.fv_label.pack(side=tk.RIGHT, padx=(0, 20))

        # Main container
        main_container = tk.Frame(page, bg=BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left — video list
        left_frame = tk.Frame(main_container, width=280, bg=BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Videos", font=("Arial", 12, "bold"),
                 bg=BG_BAR, fg=FG, pady=8).pack(fill=tk.X)

        list_container = tk.Frame(left_frame, bg=BG)
        list_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        video_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL,
                                     style="Dark.Vertical.TScrollbar")
        video_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_listbox = tk.Listbox(list_container, yscrollcommand=video_scroll.set,
                                        font=("Consolas", 10), selectmode=tk.SINGLE,
                                        bg=BG, fg=FG, selectbackground=AC_BLUE,
                                        selectforeground="white", activestyle="none",
                                        borderwidth=0, highlightthickness=0)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        video_scroll.config(command=self.video_listbox.yview)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)

        # Right — player
        right_frame = tk.Frame(main_container, bg=BG_MID)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        info_header = tk.Frame(right_frame, bg=BG_BAR, pady=8)
        info_header.pack(fill=tk.X)
        self.video_name_label = tk.Label(info_header, text="No video selected",
                                         font=("Arial", 12, "bold"), bg=BG_BAR, fg=FG)
        self.video_name_label.pack()

        meta_frame = tk.Frame(right_frame, bg=BG_MID, pady=5)
        meta_frame.pack(fill=tk.X)
        self.resolution_label = tk.Label(meta_frame, text="", font=("Arial", 9),
                                         bg=BG_MID, fg=FG_DIM)
        self.resolution_label.pack(side=tk.LEFT, padx=10)
        self.fps_label = tk.Label(meta_frame, text="", font=("Arial", 9),
                                  bg=BG_MID, fg=FG_DIM)
        self.fps_label.pack(side=tk.LEFT, padx=10)
        self.duration_label = tk.Label(meta_frame, text="", font=("Arial", 9),
                                       bg=BG_MID, fg=FG_DIM)
        self.duration_label.pack(side=tk.LEFT, padx=10)
        self.zoom_label = tk.Label(meta_frame, text="", font=("Arial", 9),
                                   bg=BG_MID, fg=FG_DIM)
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        self.yolo_btn = tk.Button(meta_frame, text="Attach YOLO video",
                                  command=self.toggle_yolo_video,
                                  font=("Arial", 9, "bold"), bg=BG_BTN2, fg=FG,
                                  padx=8, pady=1, cursor="hand2", relief=tk.FLAT, bd=0)
        self.yolo_btn.pack(side=tk.LEFT, padx=10)
        self.saved_count_label = tk.Label(meta_frame, text="", font=("Arial", 9, "bold"),
                                          bg=BG_MID, fg=AC_GREEN)
        self.saved_count_label.pack(side=tk.RIGHT, padx=10)

        # Video canvas + comment button overlay
        canvas_wrapper = tk.Frame(right_frame, bg="black")
        canvas_wrapper.pack(pady=4, padx=10)

        self.canvas = tk.Canvas(canvas_wrapper, bg="black", width=960, height=540,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self.on_zoom_reset)

        self.note_btn = tk.Button(canvas_wrapper, text="💬", command=self.open_note_popup,
                                  font=("Arial", 14), bg="#1a1a1a", fg="#888888",
                                  padx=4, pady=2, cursor="hand2",
                                  relief=tk.FLAT, bd=0, activebackground="#2a2a2a")
        self.note_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=4)

        # Slider
        slider_frame = tk.Frame(right_frame, bg=BG_MID, height=60)
        slider_frame.pack(fill=tk.X, padx=10, pady=5)
        self.slider_canvas = tk.Canvas(slider_frame, bg=BG_SLD, height=50,
                                       cursor="hand2", highlightthickness=0)
        self.slider_canvas.pack(fill=tk.X, pady=5)
        self.slider_canvas.bind("<Button-1>", self.on_slider_press)
        self.slider_canvas.bind("<B1-Motion>", self.on_slider_drag)
        self.slider_canvas.bind("<ButtonRelease-1>", self.on_slider_release)

        # Frame info
        frame_info = tk.Frame(right_frame, bg=BG_MID)
        frame_info.pack(fill=tk.X, padx=10)
        self.frame_label = tk.Label(frame_info, text="Frame: 0 / 0",
                                    font=("Arial", 10), bg=BG_MID, fg=FG)
        self.frame_label.pack(side=tk.LEFT)
        # Shortcuts popup trigger — shares the frame-label row to save height
        tk.Button(frame_info, text="?  Shortcuts & Features", command=self.show_shortcuts_popup,
                  font=("Arial", 9), bg="#2a2a2a", fg=FG_DIM,
                  padx=8, pady=1, cursor="hand2", relief=tk.FLAT, bd=0).pack(side=tk.RIGHT)

        # Controls row
        controls_frame = tk.Frame(right_frame, bg=BG_CTRL, pady=10)
        controls_frame.pack(fill=tk.X, padx=10)

        btn_style = {"font": ("Arial", 10, "bold"), "padx": 15, "pady": 8,
                     "cursor": "hand2", "relief": tk.FLAT, "bd": 0}

        tk.Button(controls_frame, text="◄◄ -10", command=lambda: self.move_frame(-10),
                  bg=BG_BTN, fg=FG, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="◄ Prev", command=lambda: self.move_frame(-1),
                  bg=BG_BTN2, fg=FG, **btn_style).pack(side=tk.LEFT, padx=2)

        self.play_btn = tk.Button(controls_frame, text="▶ Play", command=self.toggle_playback,
                                  bg=AC_BLUE, fg="white", **btn_style)
        self.play_btn.pack(side=tk.LEFT, padx=10)

        tk.Label(controls_frame, text="Speed:", font=("Arial", 10),
                 bg=BG_CTRL, fg=FG_DIM).pack(side=tk.LEFT, padx=(10, 2))
        self.speed_multiplier_var = tk.DoubleVar(value=1.0)
        self.speed_val_label = tk.Label(controls_frame, text="1.0x",
                                        font=("Arial", 9, "bold"), bg=BG_CTRL, fg=FG,
                                        width=4)
        self.speed_val_label.pack(side=tk.LEFT, padx=(0, 2))
        speed_slider = tk.Scale(controls_frame, from_=0.1, to=8.0, resolution=0.1,
                                orient=tk.HORIZONTAL, variable=self.speed_multiplier_var,
                                command=self._on_speed_slider,
                                bg=BG_CTRL, fg=FG, highlightthickness=0,
                                troughcolor="#2a2a2a", activebackground="#5a5a5a",
                                length=110, showvalue=False)
        speed_slider.pack(side=tk.LEFT)

        tk.Button(controls_frame, text="Save Frame (S)", command=self.save_current_frame,
                  bg=AC_GREEN, fg="white", **btn_style).pack(side=tk.LEFT, padx=10)
        tk.Button(controls_frame, text="Delete (Del)", command=self.delete_current_frame,
                  bg=AC_RED, fg="white", **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="Next ►", command=lambda: self.move_frame(1),
                  bg=BG_BTN2, fg=FG, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="+10 ►►", command=lambda: self.move_frame(10),
                  bg=BG_BTN, fg=FG, **btn_style).pack(side=tk.LEFT, padx=2)

        # Smart skip — same row, right side
        tk.Button(controls_frame, text="⏭ Smart Skip (E)", command=self.smart_skip,
                  bg="#4a3a6e", fg="white", **btn_style).pack(side=tk.LEFT, padx=(20, 2))

        tk.Label(controls_frame, text="Diff:",
                 font=("Arial", 9), bg=BG_CTRL, fg=FG_DIM).pack(side=tk.LEFT, padx=(8, 2))

        self.diff_threshold_var = tk.IntVar(value=4)
        self.diff_slider = tk.Scale(controls_frame, from_=1, to=7,
                                    orient=tk.HORIZONTAL, variable=self.diff_threshold_var,
                                    bg=BG_CTRL, fg=FG, highlightthickness=0,
                                    troughcolor="#2a2a2a", activebackground="#5a5a5a",
                                    length=120, showvalue=True, font=("Arial", 8))
        self.diff_slider.pack(side=tk.LEFT)

        # Status bar
        self.status_label = tk.Label(page,
                                     text=f"Ready | Save location: {self.output_folder}",
                                     bd=0, anchor=tk.W, font=("Arial", 9), bg=BG, fg=FG_DIM)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=4)

    def _fit_page(self, _event=None):
        """Stretch page to fill the window, but never below its natural size.
        Scrollbars appear only for the axis that actually overflows.

        Decision uses the outer frame (bars excluded) so showing/hiding a bar
        can't flip the result back and forth.
        """
        ow, oh = self._page_outer.winfo_width(), self._page_outer.winfo_height()
        rw, rh = self._page.winfo_reqwidth(), self._page.winfo_reqheight()
        vbar_w = self._page_vbar.winfo_reqwidth()
        hbar_h = self._page_hbar.winfo_reqheight()

        need_v = rh > oh
        need_h = rw > ow - (vbar_w if need_v else 0)
        need_v = need_v or rh > oh - (hbar_h if need_h else 0)

        if need_v:
            self._page_vbar.grid(row=0, column=1, sticky="ns")
        else:
            self._page_vbar.grid_remove()
        if need_h:
            self._page_hbar.grid(row=1, column=0, sticky="ew")
        else:
            self._page_hbar.grid_remove()

        view_w = ow - (vbar_w if need_v else 0)
        view_h = oh - (hbar_h if need_h else 0)
        w, h = max(view_w, rw), max(view_h, rh)
        self.page_canvas.itemconfig(self._page_win, width=w, height=h)
        self.page_canvas.configure(scrollregion=(0, 0, w, h))

    def _on_page_wheel(self, event):
        w = event.widget
        # Video canvas wheel = zoom, listbox wheel = list scroll; popups own their wheel
        if not isinstance(w, tk.Misc) or w.winfo_toplevel() is not self.root:
            return
        if w in (self.canvas, self.video_listbox):
            return
        step = -1 if event.delta > 0 else 1
        if event.state & 0x0001:  # Shift held → horizontal
            if self.page_canvas.xview() != (0.0, 1.0):
                self.page_canvas.xview_scroll(step, "units")
        elif self.page_canvas.yview() != (0.0, 1.0):
            self.page_canvas.yview_scroll(step, "units")

    def setup_keyboard_shortcuts(self):
        self.root.bind('a', lambda _: self.move_frame(-1))
        self.root.bind('4', lambda _: self.move_frame(-1))
        self.root.bind('6', lambda _: self.move_frame(1))
        self.root.bind('d', lambda _: self.move_frame(1))
        self.root.bind('<Left>', lambda _: self.move_frame(-1))
        self.root.bind('<Right>', lambda _: self.move_frame(1))
        self.root.bind('A', lambda _: self.move_frame(-100))
        self.root.bind('D', lambda _: self.move_frame(100))
        self.root.bind('5', lambda _: self.move_frame(50))
        self.root.bind('3', lambda _: self.move_frame(30))
        self.root.bind('2', lambda _: self.move_frame(20))
        self.root.bind('1', lambda _: self.move_frame(10))
        self.root.bind('z', lambda _: self.move_frame(-10))
        self.root.bind('s', lambda _: self.save_current_frame())
        self.root.bind('0', lambda _: self.save_current_frame())
        self.root.bind('+', lambda _: self.save_current_frame())
        self.root.bind('=', lambda _: self.save_current_frame())
        self.root.bind('q', lambda _: self.save_current_frame())
        self.root.bind('p', lambda _: self.toggle_playback())
        self.root.bind('P', lambda _: self.toggle_playback())
        self.root.bind('e', lambda _: self.smart_skip())
        self.root.bind('w', lambda _: self.navigate_video(-1))
        self.root.bind('<Up>', lambda _: self.navigate_video(-1))
        self.root.bind('<Down>', lambda _: self.navigate_video(1))
        self.root.bind('S', lambda _: self.navigate_video(1))
        self.root.bind('<Delete>', lambda _: self.delete_current_frame())

    # ──────────────────────────────────────────────────────────────────
    # Folder / video loading
    # ──────────────────────────────────────────────────────────────────

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Videos")
        if not folder:
            return
        self.video_folder = folder
        self.folder_label.config(text=Path(folder).name or folder, fg="white")

        output_folder = self.ask_output_folder()
        if output_folder is None:
            return

        self.output_folder = output_folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.output_folder_label.config(
            text=Path(self.output_folder).name or self.output_folder, fg="white")
        self.status_label.config(text=f"Ready | Save location: {self.output_folder}")
        self.load_videos()

    def ask_output_folder(self):
        today = datetime.now().strftime("%Y-%m-%d")
        default_folder = f"extracted_frames_{today}"

        dialog = tk.Toplevel(self.root)
        dialog.title("Output Folder")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.transient(self.root)
        dialog.update_idletasks()
        w, h = 380, 140
        sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        dialog.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        dialog.configure(bg="#111111")

        tk.Label(dialog, text="Where to save extracted frames?",
                 font=("Arial", 11, "bold"), pady=12, bg="#111111", fg="#c2c2c2").pack()
        tk.Label(dialog, text=f"Default: {default_folder}",
                 font=("Arial", 9), bg="#111111", fg="#5a5a5a").pack()

        choice = [None]

        def pick_default():
            choice[0] = default_folder
            dialog.destroy()

        def pick_manual():
            folder = filedialog.askdirectory(title="Select Output Folder", parent=dialog)
            if folder:
                choice[0] = folder
                dialog.destroy()

        btn_frame = tk.Frame(dialog, pady=12, bg="#111111")
        btn_frame.pack()
        tk.Button(btn_frame, text="Manual folder", command=pick_manual,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                  padx=15, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Default folder", command=pick_default,
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
                  padx=15, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=10)

        self.root.wait_window(dialog)
        return choice[0]

    def load_videos(self):
        self.videos = []
        self.yolo_map = {}  # mapping belongs to the previous folder
        self.video_listbox.delete(0, tk.END)

        all_videos = []
        for dirpath, _dirnames, filenames in os.walk(self.video_folder):
            for file in filenames:
                if file.lower().endswith(VIDEO_EXTS):
                    rel_path = os.path.relpath(
                        os.path.join(dirpath, file), self.video_folder)
                    all_videos.append(rel_path)

        for rel_path in sorted(all_videos):
            self.videos.append(rel_path)
            status = "■" if rel_path in self.frames_data.get("files", []) else "□"
            has_note = bool(self.video_notes.get(rel_path, "").strip())
            note_flag = " 💬" if has_note else ""
            self.video_listbox.insert(tk.END, f"{status} {rel_path}{note_flag}")
            if has_note:
                idx = self.video_listbox.size() - 1
                self.video_listbox.itemconfig(idx, bg="#1a3a52", fg="#7aaccc")

        self.status_label.config(
            text=f"Found {len(self.videos)} videos in {self.video_folder}")
        self.yolo_folder_btn.config(state=tk.NORMAL if self.videos else tk.DISABLED)

        if self.videos:
            self.video_listbox.selection_set(0)
            self.load_video(self.videos[0])

    def on_video_select(self, event):
        if self._updating_list:
            return
        selection = self.video_listbox.curselection()
        if selection:
            video_item = self.video_listbox.get(selection[0])[2:].removesuffix(" 💬")
            self.load_video(video_item)

    def navigate_video(self, direction):
        if not self.videos:
            return
        selection = self.video_listbox.curselection()
        if not selection:
            return
        new_idx = selection[0] + direction
        if 0 <= new_idx < len(self.videos):
            self.video_listbox.selection_clear(0, tk.END)
            self.video_listbox.selection_set(new_idx)
            self.video_listbox.see(new_idx)
            self.load_video(self.videos[new_idx])

    def load_video(self, video_name):
        self.is_playing = False
        self._scrub_gen += 1  # invalidate in-flight scrub previews of old video
        self._cached_idx, self._cached_rgb = -1, None  # cache is per-video
        self._shown_rgb = None
        old_cap = self.cap
        self.cap = None
        if self.canvas_img_id is not None:
            self.canvas.delete(self.canvas_img_id)
        self.canvas_img_id = None
        self.zoom_scale = 1.0
        self.zoom_center = (0.5, 0.5)
        self._detach_yolo()  # YOLO twin belongs to the previous video

        self.current_video = video_name
        self.current_video_path = str(Path(self.video_folder) / video_name)

        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {video_name}")
            if old_cap is not None:
                old_cap.release()
            return
        if old_cap is not None:
            old_cap.release()
        self.cap = cap

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            total = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames = max(total, 1)

        first = self._read_frame(0)
        if first is None:
            messagebox.showerror("Error", f"Cannot read first frame: {video_name}")
            return
        height, width = first.shape[:2]

        duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"

        self.video_name_label.config(text=video_name)
        self.resolution_label.config(text=f"Resolution: {width}x{height}")
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")
        self.duration_label.config(text=f"Duration: {duration_str}")
        self.zoom_label.config(text="Zoom: 1x")
        self.play_btn.config(text="▶ Play", bg="#4a9eda")

        self.current_frame_idx = 0
        self.display_frame(0)
        self.update_saved_count()
        self.draw_slider()
        self._update_note_btn()
        self.status_label.config(
            text=f"Loaded: {video_name} ({self.total_frames} frames)")
        if video_name in self.yolo_map:
            self._attach_yolo(self.yolo_map[video_name])

    # ──────────────────────────────────────────────────────────────────
    # Frame display
    # ──────────────────────────────────────────────────────────────────

    def _apply_zoom(self, frame_rgb):
        """Crop frame around zoom_center at current zoom_scale, resize to canvas."""
        if self.zoom_scale <= 1.0:
            return frame_rgb

        h, w = frame_rgb.shape[:2]
        crop_w = int(w / self.zoom_scale)
        crop_h = int(h / self.zoom_scale)

        cx = int(self.zoom_center[0] * w)
        cy = int(self.zoom_center[1] * h)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x1 = min(x1, w - crop_w)
        y1 = min(y1, h - crop_h)

        return frame_rgb[y1:y1 + crop_h, x1:x1 + crop_w]

    def _position_cap(self, frame_idx, cap=None):
        return position_cap(cap if cap is not None else self.cap, frame_idx)

    def _read_frame(self, frame_idx, cap=None):
        return read_frame(cap if cap is not None else self.cap, frame_idx)

    def _display_cap(self):
        """Cap whose frames go on screen — YOLO twin if attached, else original."""
        return self.yolo_cap if self.yolo_cap is not None else self.cap

    def _render_image(self, frame_rgb):
        """Zoom, resize and blit an RGB frame onto the canvas (main thread only)."""
        self._shown_rgb = frame_rgb
        frame_rgb = self._apply_zoom(frame_rgb)

        h, w = frame_rgb.shape[:2]
        scale = min(960 / w, 540 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image=img)

        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(480, 270, image=self.photo)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self.photo)

    def display_frame(self, frame_idx):
        if not self.cap:
            return

        frame_rgb = self._read_frame(frame_idx, self._display_cap())
        if frame_rgb is None:
            return
        self.current_frame_idx = frame_idx
        # Cache feeds save — only original pixels may land here
        if self.yolo_cap is None:
            self._cached_idx, self._cached_rgb = frame_idx, frame_rgb

        self._render_image(frame_rgb)
        self._update_frame_label(frame_idx)

        if 'handle' in self.slider_items and self.total_frames > 0:
            canvas_w = self.slider_canvas.winfo_width() or 1200
            x = (frame_idx / self.total_frames) * canvas_w
            self._move_slider_handle(x)
        else:
            self.draw_slider()

    # ──────────────────────────────────────────────────────────────────
    # Zoom
    # ──────────────────────────────────────────────────────────────────

    def on_zoom(self, event):
        if not self.cap:
            return
        # Positive delta = zoom in, negative = zoom out
        if event.num == 4 or event.delta > 0:
            self.zoom_scale = min(8.0, self.zoom_scale * 1.25)
        else:
            self.zoom_scale = max(1.0, self.zoom_scale / 1.25)

        # Update zoom center to mouse position on canvas (reset when back to 1x)
        cw, ch = 960, 540
        if self.zoom_scale > 1.0:
            self.zoom_center = (
                max(0.0, min(1.0, event.x / cw)),
                max(0.0, min(1.0, event.y / ch))
            )
        else:
            self.zoom_center = (0.5, 0.5)

        self.zoom_label.config(text=f"Zoom: {self.zoom_scale:.1f}x")
        self._rerender()

    def on_zoom_reset(self, _event):
        self.zoom_scale = 1.0
        self.zoom_center = (0.5, 0.5)
        self.zoom_label.config(text="Zoom: 1x")
        self._rerender()

    def on_canvas_click(self, event):
        if self.zoom_scale <= 1.0 or not self.cap:
            return
        self.zoom_center = (
            max(0.0, min(1.0, event.x / 960)),
            max(0.0, min(1.0, event.y / 540))
        )
        self._rerender()

    def _rerender(self):
        """Redraw the frame already on screen with current zoom — no decode."""
        if self._shown_rgb is not None:
            self._render_image(self._shown_rgb)
        else:
            self.display_frame(self.current_frame_idx)

    # ──────────────────────────────────────────────────────────────────
    # YOLO video (display-only twin, frame n ↔ frame n)
    # ──────────────────────────────────────────────────────────────────

    def toggle_yolo_video(self):
        if self.yolo_cap is not None:
            self._detach_yolo()
            self.display_frame(self.current_frame_idx)
            self.status_label.config(text="YOLO video detached — showing original")
            return
        if not self.cap or not self.current_video:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        if self.is_playing:
            self.toggle_playback()  # pause so playback doesn't tick under the dialog

        path = filedialog.askopenfilename(
            title=f"Select YOLO video for {self.current_video}",
            initialdir=self.video_folder,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        self._attach_yolo(path)

    def _attach_yolo(self, path):
        """Validate `path` against the current video and attach it. Returns success."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {Path(path).name}")
            return False

        problems = yolo_pairing.mismatch(self.total_frames, self.fps,
                                       int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                       cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if problems:
            cap.release()
            messagebox.showerror("Videos do not match",
                                 f"YOLO video not attached:\n{Path(path).name}\n\n"
                                 + "\n".join(problems))
            return False

        self.yolo_cap, self.yolo_path = cap, path
        self._scrub_gen += 1  # drop in-flight scrub frames decoded from the original
        self.yolo_btn.config(text="Detach YOLO video", bg="#d4863a", fg="white")
        self.video_name_label.config(text=f"{self.current_video}   [YOLO view]")
        self.display_frame(self.current_frame_idx)
        self.status_label.config(
            text=f"YOLO video attached: {Path(path).name} — saving still uses original")
        return True

    # ── YOLO folder: auto-pair every video by filename prefix ──

    def attach_yolo_folder(self):
        if not self.videos:
            messagebox.showwarning("Warning", "Please load a video folder first")
            return
        if self.is_playing:
            self.toggle_playback()

        folder = filedialog.askdirectory(title="Select folder containing YOLO videos",
                                         initialdir=self.video_folder)
        if not folder:
            return

        yolo_files = yolo_pairing.find_videos(folder)
        if not yolo_files:
            messagebox.showerror("Error", f"No videos found in:\n{folder}")
            return

        # YOLO folder nested inside the video folder (e.g. videos/yolo_output):
        # Browse Folder's recursive scan listed the YOLO videos as sources too.
        # They're twins, not sources — drop them from the list.
        nested = (yolo_pairing.is_under(folder, self.video_folder)
                  and not yolo_pairing.is_under(self.video_folder, folder))
        if nested:
            self.videos = [v for v in self.videos if not yolo_pairing.is_under(
                os.path.join(self.video_folder, v), folder)]

        def progress(i, n):
            self.status_label.config(text=f"Matching YOLO videos… {i}/{n}")
            self.root.update_idletasks()

        n = len(self.videos)
        self.yolo_map, failures = yolo_pairing.pair_folder(
            self.videos, self.video_folder, yolo_files, progress)

        if nested:
            self.update_video_checkmarks()
            self.update_saved_count()
        if self.current_video not in self.videos:
            # Video on screen was a YOLO twin that just got hidden — jump to first
            # source (load_video auto-attaches its twin)
            if self.videos:
                self._select_list_row(0)
                self.load_video(self.videos[0])
        else:
            if nested:
                self._select_list_row(self.videos.index(self.current_video))
            # Attach for the video on screen right now (others attach on load)
            if self.current_video in self.yolo_map:
                self._detach_yolo()
                self._attach_yolo(self.yolo_map[self.current_video])
        self.status_label.config(
            text=f"YOLO folder: {len(self.yolo_map)}/{n} videos matched"
                 + (f", {len(failures)} failed" if failures else ""))

        if failures:
            print("YOLO folder — unmatched videos:\n" + "\n".join(failures))
            shown = failures[:15]
            more = len(failures) - len(shown)
            messagebox.showwarning(
                "Some YOLO videos could not be attached",
                f"Matched {len(self.yolo_map)}/{n}. Not attached:\n\n"
                + "\n".join(shown)
                + (f"\n\n…and {more} more (full list printed to console)" if more else ""))
        else:
            messagebox.showinfo("YOLO folder", f"All {n} videos matched a YOLO video.")

    def _select_list_row(self, idx):
        self._updating_list = True
        try:
            self.video_listbox.selection_clear(0, tk.END)
            self.video_listbox.selection_set(idx)
            self.video_listbox.see(idx)
        finally:
            self._updating_list = False

    def _detach_yolo(self):
        if self.yolo_cap is None:
            return
        self.yolo_cap.release()
        self.yolo_cap, self.yolo_path = None, None
        self._scrub_gen += 1  # drop in-flight scrub frames decoded from the YOLO video
        self.yolo_btn.config(text="Attach YOLO video", bg="#2e2e2e", fg="#f0f0f0")
        if self.current_video:
            self.video_name_label.config(text=self.current_video)

    # ──────────────────────────────────────────────────────────────────
    # Shortcuts popup
    # ──────────────────────────────────────────────────────────────────

    def show_shortcuts_popup(self):
        d = tk.Toplevel(self.root)
        d.title("Shortcuts & Features")
        d.resizable(False, False)
        d.grab_set()
        d.transient(self.root)
        d.configure(bg="#111111")
        tk.Label(d, text="Shortcuts & Features", font=("Arial", 13, "bold"),
                 bg="#111111", fg="#f0f0f0", pady=12).pack()

        # Close packed before the list so it stays visible when rows overflow
        tk.Button(d, text="Close", command=d.destroy,
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
                  padx=20, pady=6, cursor="hand2", relief=tk.FLAT).pack(side=tk.BOTTOM, pady=(0, 14))

        items = [
            ("Navigation",  "A / D  or  ← →",           "Previous / Next frame"),
            ("Navigation",  "4 / 6",                     "Previous / Next frame (numpad)"),
            ("Navigation",  "Z",                         "Jump -10 frames"),
            ("Navigation",  "1",                         "Jump +10 frames"),
            ("Navigation",  "2",                         "Jump +20 frames"),
            ("Navigation",  "3",                         "Jump +30 frames"),
            ("Navigation",  "5",                         "Jump +50 frames"),
            ("Navigation",  "Shift+A / Shift+D",         "Jump ±100 frames"),
            ("Navigation",  "W / ↑",                     "Previous video"),
            ("Navigation",  "Shift+S / ↓",               "Next video"),
            ("Playback",    "p / P",                         "Play / Pause"),
            ("Playback",    "Speed menu",                "0.1x – 8x playback speed"),
            ("Save/Delete", "S  or  0  or  +  or  =  or  Q", "Save frame (auto-advances to next)"),
            ("Save/Delete", "Delete",                    "Remove saved mark for current frame"),
            ("Zoom",        "Scroll wheel",              "Zoom in / out on canvas"),
            ("Zoom",        "Click (while zoomed)",      "Re-center zoom to click position"),
            ("Zoom",        "Double-click",              "Reset zoom to 1x"),
            ("Smart Skip",  "E  or  ⏭ button",          "Jump to next meaningfully different frame"),
            ("Smart Skip",  "Diff slider",               "Scene-change sensitivity (lower = more sensitive)"),
            ("Notes",       "💬 icon (canvas top-right)", "Add / edit note for current video"),
            ("YOLO view",   "Attach / Detach button",    "Show YOLO video frame-for-frame; saves stay original"),
            ("YOLO view",   "Attach YOLO Folder (top)",  "Auto-pair every video by name prefix (_raw stripped)"),
        ]

        # Scrollable list — vertical bar shows only if rows exceed popup height
        body = tk.Frame(d, bg="#111111")
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        body_canvas = tk.Canvas(body, bg="#111111", highlightthickness=0)
        body_vbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=body_canvas.yview,
                                  style="Dark.Vertical.TScrollbar")
        body_canvas.configure(yscrollcommand=body_vbar.set)
        body_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame = tk.Frame(body_canvas, bg="#111111")
        frame_win = body_canvas.create_window(0, 0, window=frame, anchor="nw")

        def fit_body(_event=None):
            overflow = frame.winfo_reqheight() > body_canvas.winfo_height()
            if overflow and not body_vbar.winfo_ismapped():
                body_vbar.pack(side=tk.RIGHT, fill=tk.Y, before=body_canvas)
            elif not overflow and body_vbar.winfo_ismapped():
                body_vbar.pack_forget()
            body_canvas.itemconfig(frame_win, width=body_canvas.winfo_width())
            body_canvas.configure(scrollregion=(0, 0, frame.winfo_reqwidth(),
                                                frame.winfo_reqheight()))

        def wheel_body(event):
            if body_vbar.winfo_ismapped():
                body_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        frame.bind("<Configure>", fit_body)
        body_canvas.bind("<Configure>", fit_body)
        d.bind("<MouseWheel>", wheel_body)  # toplevel tag → fires for every child

        current_cat = None
        for cat, key, desc in items:
            if cat != current_cat:
                current_cat = cat
                tk.Label(frame, text=cat, font=("Arial", 9, "bold"),
                         bg="#111111", fg="#4a9eda", anchor="w").pack(fill=tk.X, pady=(8, 1))
            row = tk.Frame(frame, bg="#1a1a1a")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, font=("Consolas", 9), width=26, anchor="w",
                     bg="#1a1a1a", fg="#d0d0d0", padx=6).pack(side=tk.LEFT)
            tk.Label(row, text=desc, font=("Arial", 9), anchor="w",
                     bg="#1a1a1a", fg="#9a9a9a", padx=6).pack(side=tk.LEFT)

        # Size to content (swap canvas default height for the rows' real
        # height), clamped to screen — bar appears only when clamped
        d.update_idletasks()
        sw, sh = d.winfo_screenwidth(), d.winfo_screenheight()
        need_h = d.winfo_reqheight() - body_canvas.winfo_reqheight() + frame.winfo_reqheight()
        w, h = 540, min(need_h, sh - 90)
        d.geometry(f"{w}x{h}+{(sw-w)//2}+{max(0, (sh-h)//2)}")

    # ──────────────────────────────────────────────────────────────────
    # Video notes
    # ──────────────────────────────────────────────────────────────────

    def _update_note_btn(self):
        if not self.current_video:
            return
        has_note = bool(self.video_notes.get(self.current_video, "").strip())
        self.note_btn.config(fg="#4a9eda" if has_note else "#888888")

    def open_note_popup(self):
        if not self.current_video:
            return
        existing = self.video_notes.get(self.current_video, "")

        d = tk.Toplevel(self.root)
        d.title(f"Note — {self.current_video}")
        d.resizable(False, False)
        d.grab_set()
        d.transient(self.root)
        d.configure(bg="#111111")
        d.update_idletasks()
        w, h = 480, 280
        sw, sh = d.winfo_screenwidth(), d.winfo_screenheight()
        d.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        tk.Label(d, text=self.current_video, font=("Arial", 10, "bold"),
                 bg="#111111", fg="#7aaccc", pady=10, wraplength=460).pack()

        txt = tk.Text(d, font=("Arial", 10), bg="#1e1e1e", fg="#f0f0f0",
                      insertbackground="white", relief=tk.FLAT,
                      padx=8, pady=8, wrap=tk.WORD, height=8)
        txt.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))
        txt.insert("1.0", existing)
        txt.focus_set()

        def save_note():
            note = txt.get("1.0", tk.END).strip()
            if note:
                self.video_notes[self.current_video] = note
            elif self.current_video in self.video_notes:
                del self.video_notes[self.current_video]
            self.frames_data["notes"] = self.video_notes
            self.save_session()
            self._update_note_btn()
            self.update_video_checkmarks()
            d.destroy()

        btn_frame = tk.Frame(d, bg="#111111")
        btn_frame.pack(pady=(0, 12))
        tk.Button(btn_frame, text="Save", command=save_note,
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
                  padx=16, pady=5, cursor="hand2", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Close", command=d.destroy,
                  bg="#444444", fg="white", font=("Arial", 10, "bold"),
                  padx=16, pady=5, cursor="hand2", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)

    # ──────────────────────────────────────────────────────────────────
    # Smart skip
    # ──────────────────────────────────────────────────────────────────

    def smart_skip(self):
        if not self.cap or self.total_frames == 0:
            return

        THUMB = (160, 90)
        STRIDE = 5
        threshold = self.diff_threshold_var.get()
        start = self.current_frame_idx

        start_full = self._read_frame(start)
        if start_full is None:
            self.status_label.config(text="Smart skip: cannot read reference frame")
            return
        ref = cv2.cvtColor(
            cv2.resize(start_full, THUMB, interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY).astype(np.float32)

        self.status_label.config(text="Scanning for next meaningful frame...")
        self.root.update_idletasks()

        # sequential read: seek once to start+1, read every frame, compare every STRIDE-th
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start + 1)
        current = start + 1
        for idx in range(start + STRIDE, self.total_frames, STRIDE):
            # skip frames between last read position and target
            while current < idx:
                ret, _ = self.cap.read()
                if not ret:
                    self.status_label.config(text="Smart skip: no significant change found")
                    return
                current += 1
            ret, bgr = self.cap.read()
            if not ret:
                break
            current += 1
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            curr = cv2.cvtColor(
                cv2.resize(frame_rgb, THUMB, interpolation=cv2.INTER_AREA),
                cv2.COLOR_RGB2GRAY).astype(np.float32)
            diff = np.mean(np.abs(curr - ref))
            if diff >= threshold:
                    self.display_frame(idx)
                    self.draw_slider()
                    self.status_label.config(
                        text=f"Smart skip → frame {idx} (diff={diff:.1f})")
                    return

        self.status_label.config(text="Smart skip: no significant change found")

    # ──────────────────────────────────────────────────────────────────
    # Playback
    # ──────────────────────────────────────────────────────────────────

    def _playback_params(self):
        """Returns (delay_ms, frames_to_advance).

        Tkinter after() on Windows has ~16ms minimum resolution.
        At high speeds the ideal delay is sub-16ms, so we compensate by
        advancing multiple frames per tick instead of hammering the timer.
        """
        fps = self.fps if self.fps > 0 else 25
        multiplier = max(0.1, self.speed_multiplier_var.get())
        ideal_delay = 1000 / fps / multiplier
        delay_ms = max(16, int(ideal_delay))
        frames = max(1, round(delay_ms * fps * multiplier / 1000))
        return delay_ms, frames

    def play_frames(self):
        if not self.is_playing:
            return

        delay_ms, frames_advance = self._playback_params()
        next_idx = self.current_frame_idx + frames_advance
        if next_idx >= self.total_frames:
            self.is_playing = False
            self.play_btn.config(text="▶ Play", bg="#4a9eda")
            self.draw_slider()
            self.status_label.config(text="Playback finished")
            return

        # Grab-forward for short advances, real seek only when out of sync
        cap = self._display_cap()
        self._position_cap(next_idx, cap)

        ret, bgr = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.current_frame_idx = next_idx
            if self.yolo_cap is None:
                self._cached_idx, self._cached_rgb = next_idx, frame_rgb
            self._render_image(frame_rgb)
            self._update_frame_label(next_idx)
            if next_idx % 10 == 0:
                self.draw_slider()
            elif 'handle' in self.slider_items and self.total_frames > 0:
                canvas_w = self.slider_canvas.winfo_width() or 1200
                x = (next_idx / self.total_frames) * canvas_w
                self._move_slider_handle(x)

        self.root.after(delay_ms, self.play_frames)

    def move_frame(self, delta):
        if not self.cap:
            return
        new_idx = max(0, min(self.current_frame_idx + delta, self.total_frames - 1))
        self.display_frame(new_idx)

    def toggle_playback(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸ Pause", bg="#d4863a")
            self.root.after(self._playback_params()[0], self.play_frames)
        else:
            self.play_btn.config(text="▶ Play", bg="#4a9eda")
            self.draw_slider()

    def _on_speed_slider(self, *_):
        v = self.speed_multiplier_var.get()
        self.speed_val_label.config(text=f"{v:.1f}x")

    # ──────────────────────────────────────────────────────────────────
    # Slider
    # ──────────────────────────────────────────────────────────────────

    def on_slider_press(self, event):
        if not self.cap or self.total_frames == 0:
            return
        self.slider_dragging = True
        self._scrub_to(event.x)

    def on_slider_drag(self, event):
        if not self.cap or self.total_frames == 0 or not self.slider_dragging:
            return
        self._scrub_to(event.x)

    def on_slider_release(self, _event):
        self.slider_dragging = False
        if self._slider_after_id is not None:
            self.root.after_cancel(self._slider_after_id)
            self._slider_after_id = None
        # Authoritative exact decode on the main cap once drag ends
        self.display_frame(self.current_frame_idx)

    def _scrub_to(self, x):
        """Handle + label move instantly; frame decode goes to the worker."""
        canvas_width = self.slider_canvas.winfo_width()
        if canvas_width <= 0:
            return
        x = max(0, min(x, canvas_width))
        frame_idx = max(0, min(int((x / canvas_width) * self.total_frames),
                               self.total_frames - 1))
        self.current_frame_idx = frame_idx
        self._move_slider_handle(x)
        self._update_frame_label(frame_idx)
        self._request_scrub_frame(frame_idx)

    # ──────────────────────────────────────────────────────────────────
    # Scrub worker
    # ──────────────────────────────────────────────────────────────────

    def _request_scrub_frame(self, frame_idx):
        if not self.current_video_path:
            return
        path = self.yolo_path or self.current_video_path
        self._scrub.request(path, frame_idx, self._scrub_gen)

    def _on_scrub_result(self, frame_rgb, idx, gen):
        if gen != self._scrub_gen:
            return  # stale result from a previously loaded video / YOLO state
        if self.yolo_cap is None:
            self._cached_idx, self._cached_rgb = idx, frame_rgb
        self._render_image(frame_rgb)

    def _move_slider_handle(self, x_pos):
        if 'handle' not in self.slider_items:
            self.draw_slider()
            return
        self.slider_canvas.coords(self.slider_items['handle'],
                                   x_pos - 3, 15, x_pos + 3, 35)
        saved_frames = self.get_saved_frames()
        if self.current_frame_idx in saved_frames:
            if 'active_dot' not in self.slider_items:
                self.slider_items['active_dot'] = self.slider_canvas.create_oval(
                    x_pos - 6, 10, x_pos + 6, 22, fill="#27ae60", outline="")
            else:
                self.slider_canvas.coords(self.slider_items['active_dot'],
                                           x_pos - 6, 10, x_pos + 6, 22)
        elif 'active_dot' in self.slider_items:
            self.slider_canvas.delete(self.slider_items.pop('active_dot'))

    def _update_frame_label(self, frame_idx):
        time_sec = frame_idx / self.fps if self.fps > 0 else 0
        self.frame_label.config(
            text=f"Frame: {frame_idx} / {self.total_frames} | "
                 f"Time: {int(time_sec // 60)}:{time_sec % 60:05.2f}")

    def draw_slider(self):
        self.slider_canvas.delete("all")
        self.slider_items.clear()

        if not self.cap or self.total_frames == 0:
            return

        canvas_width = self.slider_canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 1200

        self.slider_canvas.create_rectangle(0, 20, canvas_width, 30,
                                             fill="#333333", outline="")

        saved_frames = self.get_saved_frames()
        for frame_idx in saved_frames:
            x = (frame_idx / self.total_frames) * canvas_width
            self.slider_canvas.create_rectangle(x - 1, 20, x + 1, 30,
                                                 fill="#27ae60", outline="")

        current_x = (self.current_frame_idx / self.total_frames) * canvas_width
        self.slider_items['handle'] = self.slider_canvas.create_rectangle(
            current_x - 3, 15, current_x + 3, 35,
            fill="#3498db", outline="#2980b9", width=2)

        if self.current_frame_idx in saved_frames:
            self.slider_items['active_dot'] = self.slider_canvas.create_oval(
                current_x - 6, 10, current_x + 6, 22, fill="#27ae60", outline="")

    # ──────────────────────────────────────────────────────────────────
    # Save / delete
    # ──────────────────────────────────────────────────────────────────

    def save_current_frame(self):
        if not self.cap or not self.current_video:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        video_output_folder = os.path.join(self.output_folder, video_name)
        Path(video_output_folder).mkdir(exist_ok=True)

        filename = f"{video_name}_frame_{self.current_frame_idx:06d}.jpg"
        filepath = os.path.join(video_output_folder, filename)

        if os.path.exists(filepath):
            self.status_label.config(text=f"Already saved: {filename}")
            return

        # Reuse the frame already decoded for display — re-reading would
        # cost a full keyframe seek + forward decode
        if self._cached_idx == self.current_frame_idx and self._cached_rgb is not None:
            frame_rgb = self._cached_rgb
        else:
            frame_rgb = self._read_frame(self.current_frame_idx)
        if frame_rgb is None:
            self.status_label.config(text="Failed to read frame")
            return
        cv2.imwrite(filepath, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if self.current_video not in self.frames_data:
            self.frames_data[self.current_video] = []
        if self.current_frame_idx not in self.frames_data[self.current_video]:
            self.frames_data[self.current_video].append(self.current_frame_idx)
            self.frames_data[self.current_video].sort()
        if self.current_video not in self.frames_data.get("files", []):
            self.frames_data.setdefault("files", []).append(self.current_video)

        self.save_session()
        self.update_saved_count()
        self.update_video_checkmarks()
        # Full redraw so the green dot appears on the saved frame before advancing
        self.draw_slider()

        if not self.is_playing:
            self.move_frame(1)

        self.status_label.config(text=f"✓ Saved: {filename}")

    def delete_current_frame(self):
        if not self.cap or not self.current_video:
            return

        saved_frames = self.get_saved_frames()
        if self.current_frame_idx not in saved_frames:
            self.status_label.config(text="Frame not saved, nothing to delete")
            return

        video_name = Path(self.current_video).stem
        video_output_folder = os.path.join(self.output_folder, video_name)
        filename = f"{video_name}_frame_{self.current_frame_idx:06d}.jpg"
        filepath = os.path.join(video_output_folder, filename)

        if os.path.exists(filepath):
            os.remove(filepath)

        self.frames_data[self.current_video].remove(self.current_frame_idx)
        if not self.frames_data[self.current_video]:
            del self.frames_data[self.current_video]
            if self.current_video in self.frames_data.get("files", []):
                self.frames_data["files"].remove(self.current_video)

        self.save_session()
        self.update_saved_count()
        self.update_video_checkmarks()
        self.draw_slider()
        self.status_label.config(text=f"✗ Deleted: {filename}")

    # ──────────────────────────────────────────────────────────────────
    # Session / helpers
    # ──────────────────────────────────────────────────────────────────

    def get_saved_frames(self):
        if not self.current_video:
            return []
        return self.frames_data.get(self.current_video, [])

    def _get_session_totals(self):
        # Scope totals to videos of the currently loaded folder only —
        # session JSON may hold entries from other input folders
        total_frames = sum(len(self.frames_data.get(v, [])) for v in self.videos)
        total_videos = sum(1 for v in self.videos
                           if v in self.frames_data.get("files", []))
        return total_frames, total_videos

    def update_saved_count(self):
        self.saved_count_label.config(
            text=f"Saved frames: {len(self.get_saved_frames())}")
        total_f, total_v = self._get_session_totals()
        self.fv_label.config(text=f"F/V: {total_f}/{total_v}")

    def update_video_checkmarks(self):
        if not self.video_folder:
            return
        selection = self.video_listbox.curselection()
        current_idx = selection[0] if selection else None
        scroll_pos = self.video_listbox.yview()[0]

        self._updating_list = True
        try:
            self.video_listbox.delete(0, tk.END)
            for video in self.videos:
                status = "■" if video in self.frames_data.get("files", []) else "□"
                has_note = bool(self.video_notes.get(video, "").strip())
                note_flag = " 💬" if has_note else ""
                self.video_listbox.insert(tk.END, f"{status} {video}{note_flag}")
                if has_note:
                    idx = self.video_listbox.size() - 1
                    self.video_listbox.itemconfig(idx, bg="#1a3a52", fg="#7aaccc")
            if current_idx is not None:
                self.video_listbox.selection_set(current_idx)
        finally:
            self._updating_list = False
        self.video_listbox.yview_moveto(scroll_pos)

    def save_session(self):
        session.save_session(self.frames_data, self.session_file)

    def load_session(self):
        self.frames_data = session.load_session(self.session_file)

    def on_closing(self):
        self.is_playing = False
        if self._slider_after_id is not None:
            self.root.after_cancel(self._slider_after_id)
            self._slider_after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._detach_yolo()
        self.save_session()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FrameReviewApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
