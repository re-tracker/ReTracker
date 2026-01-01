"""Interactive streaming tracking pipeline with UI controls."""

import time
import cv2
import numpy as np
import torch
import imageio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass

from ..config import StreamingConfig
from .streaming_source import StreamingSourceFactory
from .query_generator import QueryGeneratorFactory
from .tracker import Tracker
from retracker.utils.rich_utils import CONSOLE


@dataclass
class KeyframeConfig:
    """Configuration for automatic keyframe detection."""
    # Minimum frame interval between keyframes
    min_interval: int = 10
    # Minimum median parallax (pixels) to trigger keyframe
    min_parallax: float = 15.0
    # Minimum visible ratio to trigger keyframe (below this = create keyframe)
    # This is now relative to target_points: threshold = min_visible_ratio * target_points
    min_visible_ratio: float = 0.2


@dataclass
class UIState:
    """State container for UI controls."""
    is_paused: bool = False
    should_add_keyframe: bool = False
    should_quit: bool = False
    # Keyframe mode: 'auto' or 'manual'
    keyframe_mode: str = 'auto'
    # Show trajectory trace on previous frames
    show_trace: bool = False
    # Number of points to track (can be adjusted dynamically)
    num_points: int = 64
    # Flag to regenerate queries with new point count
    should_regenerate_queries: bool = False
    # Center crop settings
    enable_crop: bool = False
    crop_ratio: float = 1.0  # 1.0 = no crop, 0.5 = crop to 50% center
    # Real-time mode: True = process latest frame, False = process sequentially
    realtime_mode: bool = False
    # Video seek control
    seek_to_frame: int = -1  # -1 means no seek requested
    total_frames: int = 0  # Total frames in video (0 if unknown)
    current_frame: int = 0  # Current frame position
    # Point selection mode - for manually adding tracking points
    point_selection_mode: bool = False  # True = in selection mode (video paused, click to add points)
    selected_points: List[Tuple[float, float]] = None  # Manually selected points [(x, y), ...]
    should_finish_selection: bool = False  # Flag to finish selection and create keyframe
    # Lock for thread-safe access
    lock: threading.Lock = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()
        if self.selected_points is None:
            self.selected_points = []

    def set_paused(self, paused: bool):
        with self.lock:
            self.is_paused = paused

    def get_paused(self) -> bool:
        with self.lock:
            return self.is_paused

    def trigger_add_keyframe(self):
        with self.lock:
            self.should_add_keyframe = True

    def consume_add_keyframe(self) -> bool:
        with self.lock:
            if self.should_add_keyframe:
                self.should_add_keyframe = False
                return True
            return False

    def set_quit(self):
        with self.lock:
            self.should_quit = True

    def get_quit(self) -> bool:
        with self.lock:
            return self.should_quit

    def set_keyframe_mode(self, mode: str):
        """Set keyframe mode: 'auto' or 'manual'."""
        with self.lock:
            self.keyframe_mode = mode

    def get_keyframe_mode(self) -> str:
        with self.lock:
            return self.keyframe_mode

    def is_auto_keyframe(self) -> bool:
        with self.lock:
            return self.keyframe_mode == 'auto'

    def set_show_trace(self, show: bool):
        """Set whether to show trajectory trace."""
        with self.lock:
            self.show_trace = show

    def get_show_trace(self) -> bool:
        with self.lock:
            return self.show_trace

    def toggle_show_trace(self) -> bool:
        """Toggle show trace and return new state."""
        with self.lock:
            self.show_trace = not self.show_trace
            return self.show_trace

    def get_num_points(self) -> int:
        with self.lock:
            return self.num_points

    def set_num_points(self, n: int):
        with self.lock:
            self.num_points = max(8, min(512, n))  # Clamp between 8 and 512

    def increase_points(self, factor: float = 2.0) -> int:
        """Increase point count and return new value."""
        with self.lock:
            self.num_points = min(512, int(self.num_points * factor))
            self.should_regenerate_queries = True
            return self.num_points

    def decrease_points(self, factor: float = 2.0) -> int:
        """Decrease point count and return new value."""
        with self.lock:
            self.num_points = max(8, int(self.num_points / factor))
            self.should_regenerate_queries = True
            return self.num_points

    def consume_regenerate_queries(self) -> bool:
        """Check and consume regenerate queries flag."""
        with self.lock:
            if self.should_regenerate_queries:
                self.should_regenerate_queries = False
                return True
            return False

    def set_enable_crop(self, enable: bool):
        """Set whether center crop is enabled."""
        with self.lock:
            self.enable_crop = enable

    def get_enable_crop(self) -> bool:
        with self.lock:
            return self.enable_crop

    def set_crop_ratio(self, ratio: float):
        """Set crop ratio (0.1 to 1.0)."""
        with self.lock:
            self.crop_ratio = max(0.1, min(1.0, ratio))

    def get_crop_ratio(self) -> float:
        with self.lock:
            return self.crop_ratio

    def set_realtime_mode(self, enabled: bool):
        """Set real-time mode (True = process latest frame, False = sequential)."""
        with self.lock:
            self.realtime_mode = enabled

    def get_realtime_mode(self) -> bool:
        with self.lock:
            return self.realtime_mode

    def toggle_realtime_mode(self) -> bool:
        """Toggle real-time mode and return new state."""
        with self.lock:
            self.realtime_mode = not self.realtime_mode
            return self.realtime_mode

    def set_total_frames(self, total: int):
        """Set total number of frames in video."""
        with self.lock:
            self.total_frames = total

    def get_total_frames(self) -> int:
        with self.lock:
            return self.total_frames

    def set_current_frame(self, frame: int):
        """Set current frame position."""
        with self.lock:
            self.current_frame = frame

    def get_current_frame(self) -> int:
        with self.lock:
            return self.current_frame

    def request_seek(self, frame: int):
        """Request to seek to a specific frame."""
        with self.lock:
            self.seek_to_frame = frame

    def consume_seek_request(self) -> int:
        """Get and clear seek request. Returns -1 if no request."""
        with self.lock:
            frame = self.seek_to_frame
            self.seek_to_frame = -1
            return frame

    def has_seek_request(self) -> bool:
        with self.lock:
            return self.seek_to_frame >= 0

    # Point selection methods
    def start_point_selection(self):
        """Start point selection mode - pauses video and enables clicking."""
        with self.lock:
            self.point_selection_mode = True
            self.is_paused = True
            self.selected_points.clear()

    def stop_point_selection(self):
        """Stop point selection mode without applying."""
        with self.lock:
            self.point_selection_mode = False
            self.selected_points.clear()

    def get_point_selection_mode(self) -> bool:
        with self.lock:
            return self.point_selection_mode

    def add_selected_point(self, x: float, y: float):
        """Add a point to the selection list."""
        with self.lock:
            if self.point_selection_mode:
                self.selected_points.append((x, y))

    def remove_nearby_point(self, x: float, y: float, threshold: float = 15.0) -> bool:
        """Remove the nearest point within threshold. Returns True if a point was removed."""
        with self.lock:
            if not self.selected_points:
                return False
            # Find nearest point
            min_dist = float('inf')
            min_idx = -1
            for i, (px, py) in enumerate(self.selected_points):
                dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_dist < threshold and min_idx >= 0:
                self.selected_points.pop(min_idx)
                return True
            return False

    def get_selected_points(self) -> List[Tuple[float, float]]:
        """Get copy of currently selected points."""
        with self.lock:
            return list(self.selected_points)

    def clear_selected_points(self):
        """Clear all selected points."""
        with self.lock:
            self.selected_points.clear()

    def get_selected_point_count(self) -> int:
        """Get total number of selected points."""
        with self.lock:
            return len(self.selected_points)

    def trigger_finish_selection(self):
        """Trigger finishing selection and creating keyframe with selected + visible points."""
        with self.lock:
            self.should_finish_selection = True

    def consume_finish_selection(self) -> bool:
        """Check and consume finish selection flag."""
        with self.lock:
            if self.should_finish_selection:
                self.should_finish_selection = False
                return True
            return False


def has_display() -> bool:
    """Check if a display is available (X11/Wayland)."""
    import os
    display = os.environ.get('DISPLAY', '')
    wayland = os.environ.get('WAYLAND_DISPLAY', '')
    return bool(display or wayland)


class StreamingControlUI:
    """Tkinter-based control panel for streaming pipeline."""

    def __init__(self, ui_state: UIState):
        self.ui_state = ui_state
        self.root = None
        self.thread = None
        self._display_available = has_display()

    def _create_ui(self):
        """Create the tkinter UI (runs in separate thread)."""
        if not self._display_available:
            CONSOLE.print("[yellow][UI][/yellow] No display available, UI disabled")
            return

        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title("Streaming Tracker Control")
        self.root.geometry("320x850")
        self.root.resizable(False, False)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0

        # Status label
        self.status_var = tk.StringVar(value="Status: Running")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Helvetica', 12))
        status_label.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        # ========== Video Progress Section (only shown for video sources) ==========
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1

        # Frame position label
        self.frame_pos_var = tk.StringVar(value="Frame: 0 / 0")
        frame_pos_label = ttk.Label(self.progress_frame, textvariable=self.frame_pos_var, font=('Helvetica', 9))
        frame_pos_label.pack(pady=2)

        # Progress slider
        self.progress_var = tk.IntVar(value=0)
        self.progress_slider = ttk.Scale(
            self.progress_frame, from_=0, to=100,
            variable=self.progress_var, orient=tk.HORIZONTAL,
            length=280, command=self._on_progress_change
        )
        self.progress_slider.pack(pady=2)

        # Seek button row
        seek_btn_frame = ttk.Frame(self.progress_frame)
        seek_btn_frame.pack(pady=2)

        self.seek_back_btn = ttk.Button(seek_btn_frame, text="<< -100", width=8, command=lambda: self._seek_relative(-100))
        self.seek_back_btn.pack(side=tk.LEFT, padx=2)

        self.seek_back_small_btn = ttk.Button(seek_btn_frame, text="< -10", width=6, command=lambda: self._seek_relative(-10))
        self.seek_back_small_btn.pack(side=tk.LEFT, padx=2)

        self.seek_fwd_small_btn = ttk.Button(seek_btn_frame, text="+10 >", width=6, command=lambda: self._seek_relative(10))
        self.seek_fwd_small_btn.pack(side=tk.LEFT, padx=2)

        self.seek_fwd_btn = ttk.Button(seek_btn_frame, text="+100 >>", width=8, command=lambda: self._seek_relative(100))
        self.seek_fwd_btn.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # Pause/Resume button
        self.pause_button_text = tk.StringVar(value="Pause (Space)")
        self.pause_button = ttk.Button(
            main_frame,
            textvariable=self.pause_button_text,
            command=self._toggle_pause,
            width=18
        )
        self.pause_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1

        # ========== Points Control Section ==========
        points_label = ttk.Label(main_frame, text="Track Points:", font=('Helvetica', 10, 'bold'))
        points_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Points display and buttons
        points_frame = ttk.Frame(main_frame)
        points_frame.grid(row=row, column=0, columnspan=2, pady=5)

        self.points_minus_btn = ttk.Button(points_frame, text="-", width=3, command=self._decrease_points)
        self.points_minus_btn.pack(side=tk.LEFT, padx=5)

        self.points_var = tk.StringVar(value=str(self.ui_state.get_num_points()))
        points_display = ttk.Label(points_frame, textvariable=self.points_var, font=('Helvetica', 14, 'bold'), width=5, anchor='center')
        points_display.pack(side=tk.LEFT, padx=10)

        self.points_plus_btn = ttk.Button(points_frame, text="+", width=3, command=self._increase_points)
        self.points_plus_btn.pack(side=tk.LEFT, padx=5)
        row += 1

        # Regenerate button
        self.regen_button = ttk.Button(
            main_frame,
            text="Regenerate Points (R)",
            command=self._regenerate_points,
            width=20
        )
        self.regen_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1

        # ========== Keyframe Mode Section ==========
        kf_mode_label = ttk.Label(main_frame, text="Keyframe Mode:", font=('Helvetica', 10, 'bold'))
        kf_mode_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Auto/Manual radio buttons
        self.keyframe_mode_var = tk.StringVar(value=self.ui_state.get_keyframe_mode())

        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=row, column=0, columnspan=2, pady=5)

        auto_radio = ttk.Radiobutton(
            mode_frame, text="Auto", variable=self.keyframe_mode_var,
            value='auto', command=self._on_mode_change
        )
        auto_radio.pack(side=tk.LEFT, padx=10)

        manual_radio = ttk.Radiobutton(
            mode_frame, text="Manual", variable=self.keyframe_mode_var,
            value='manual', command=self._on_mode_change
        )
        manual_radio.pack(side=tk.LEFT, padx=10)
        row += 1

        # Add Keyframe button
        self.keyframe_button = ttk.Button(
            main_frame, text="Add Keyframe (K)",
            command=self._add_keyframe, width=18
        )
        self.keyframe_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # Show Trace checkbox
        self.show_trace_var = tk.BooleanVar(value=self.ui_state.get_show_trace())
        self.trace_checkbox = ttk.Checkbutton(
            main_frame, text="Show Trajectory Trace (T)",
            variable=self.show_trace_var, command=self._on_trace_toggle
        )
        self.trace_checkbox.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # ========== Center Crop Section ==========
        crop_label = ttk.Label(main_frame, text="Center Crop:", font=('Helvetica', 10, 'bold'))
        crop_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Enable crop checkbox
        self.enable_crop_var = tk.BooleanVar(value=self.ui_state.get_enable_crop())
        self.crop_checkbox = ttk.Checkbutton(
            main_frame, text="Enable Center Crop",
            variable=self.enable_crop_var, command=self._on_crop_toggle
        )
        self.crop_checkbox.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Crop ratio slider
        crop_slider_frame = ttk.Frame(main_frame)
        crop_slider_frame.grid(row=row, column=0, columnspan=2, pady=5)

        ttk.Label(crop_slider_frame, text="Ratio:").pack(side=tk.LEFT, padx=2)

        self.crop_ratio_var = tk.DoubleVar(value=self.ui_state.get_crop_ratio())
        self.crop_slider = ttk.Scale(
            crop_slider_frame, from_=0.2, to=1.0,
            variable=self.crop_ratio_var, orient=tk.HORIZONTAL,
            length=150, command=self._on_crop_ratio_change
        )
        self.crop_slider.pack(side=tk.LEFT, padx=5)

        self.crop_ratio_label = ttk.Label(crop_slider_frame, text=f"{self.crop_ratio_var.get():.1%}", width=5)
        self.crop_ratio_label.pack(side=tk.LEFT, padx=2)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # ========== Processing Mode Section ==========
        mode_label = ttk.Label(main_frame, text="Processing Mode:", font=('Helvetica', 10, 'bold'))
        mode_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Real-time mode checkbox
        self.realtime_var = tk.BooleanVar(value=self.ui_state.get_realtime_mode())
        self.realtime_checkbox = ttk.Checkbutton(
            main_frame, text="Real-time Mode (L)",
            variable=self.realtime_var, command=self._on_realtime_toggle
        )
        self.realtime_checkbox.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Mode description
        self.realtime_desc_var = tk.StringVar(value=self._get_realtime_description())
        realtime_desc = ttk.Label(main_frame, textvariable=self.realtime_desc_var,
                                  font=('Helvetica', 8), foreground='gray')
        realtime_desc.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # ========== Manual Point Selection Section ==========
        point_sel_label = ttk.Label(main_frame, text="Manual Point Selection:", font=('Helvetica', 10, 'bold'))
        point_sel_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Selected points count
        self.selected_count_var = tk.StringVar(value="")
        selected_count_label = ttk.Label(main_frame, textvariable=self.selected_count_var,
                                         font=('Helvetica', 9), foreground='blue')
        selected_count_label.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Point selection buttons - Start/Done
        point_btn_frame = ttk.Frame(main_frame)
        point_btn_frame.grid(row=row, column=0, columnspan=2, pady=5)

        self.start_sel_btn = ttk.Button(
            point_btn_frame, text="Start Selection (P)",
            command=self._start_point_selection, width=16
        )
        self.start_sel_btn.pack(side=tk.LEFT, padx=2)

        self.done_sel_btn = ttk.Button(
            point_btn_frame, text="Done (Enter)",
            command=self._finish_point_selection, width=12,
            state='disabled'
        )
        self.done_sel_btn.pack(side=tk.LEFT, padx=2)
        row += 1

        # Cancel button
        self.cancel_sel_btn = ttk.Button(
            main_frame, text="Cancel (ESC)",
            command=self._cancel_point_selection, width=15,
            state='disabled'
        )
        self.cancel_sel_btn.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Point selection help text
        self.point_help_var = tk.StringVar(value="Pause video, click to add points, then Done")
        point_help_label = ttk.Label(main_frame, textvariable=self.point_help_var,
                                     font=('Helvetica', 8), foreground='gray')
        point_help_label.grid(row=row, column=0, columnspan=2, pady=2)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # Status message
        self.status_msg_var = tk.StringVar(value="")
        status_msg_label = ttk.Label(main_frame, textvariable=self.status_msg_var, foreground='green')
        status_msg_label.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Quit button
        quit_button = ttk.Button(main_frame, text="Quit (Q)", command=self._quit, width=15)
        quit_button.grid(row=row, column=0, columnspan=2, pady=10)

        # Keyboard bindings
        self.root.bind('<space>', lambda e: self._toggle_pause())
        self.root.bind('<k>', lambda e: self._add_keyframe())
        self.root.bind('<K>', lambda e: self._add_keyframe())
        self.root.bind('<m>', lambda e: self._toggle_keyframe_mode())
        self.root.bind('<M>', lambda e: self._toggle_keyframe_mode())
        self.root.bind('<t>', lambda e: self._toggle_trace())
        self.root.bind('<T>', lambda e: self._toggle_trace())
        self.root.bind('<plus>', lambda e: self._increase_points())
        self.root.bind('<equal>', lambda e: self._increase_points())
        self.root.bind('<minus>', lambda e: self._decrease_points())
        self.root.bind('<r>', lambda e: self._regenerate_points())
        self.root.bind('<R>', lambda e: self._regenerate_points())
        self.root.bind('<l>', lambda e: self._toggle_realtime())
        self.root.bind('<L>', lambda e: self._toggle_realtime())
        self.root.bind('<p>', lambda e: self._start_point_selection())
        self.root.bind('<P>', lambda e: self._start_point_selection())
        self.root.bind('<q>', lambda e: self._quit())
        self.root.bind('<Escape>', lambda e: self._cancel_or_quit())

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # Start the main loop
        self.root.mainloop()

    def _get_mode_description(self) -> str:
        """Get description text for current keyframe mode."""
        mode = self.keyframe_mode_var.get() if hasattr(self, 'keyframe_mode_var') else self.ui_state.get_keyframe_mode()
        if mode == 'auto':
            return "Auto: Keyframes created based on parallax, visible ratio, and frame interval"
        else:
            return "Manual: Keyframes only created when you click 'Add Keyframe' or press K"

    def _on_mode_change(self):
        """Handle keyframe mode change."""
        mode = self.keyframe_mode_var.get()
        self.ui_state.set_keyframe_mode(mode)
        self.mode_desc_var.set(self._get_mode_description())
        self.status_msg_var.set(f"Mode: {mode.upper()}")
        if self.root:
            self.root.after(1500, lambda: self.status_msg_var.set(""))

    def _toggle_keyframe_mode(self):
        """Toggle between auto and manual keyframe mode."""
        current = self.ui_state.get_keyframe_mode()
        new_mode = 'manual' if current == 'auto' else 'auto'
        self.keyframe_mode_var.set(new_mode)
        self._on_mode_change()

    def _toggle_pause(self):
        """Toggle pause state."""
        current = self.ui_state.get_paused()
        self.ui_state.set_paused(not current)

        if not current:
            self.status_var.set("Status: Paused")
            self.pause_button_text.set("Resume")
        else:
            self.status_var.set("Status: Running")
            self.pause_button_text.set("Pause")

    def _add_keyframe(self):
        """Request to add a keyframe at current frame."""
        self.ui_state.trigger_add_keyframe()
        self.status_msg_var.set("Keyframe requested!")
        # Clear the message after 1 second
        if self.root:
            self.root.after(1000, lambda: self.status_msg_var.set(""))

    def _on_trace_toggle(self):
        """Handle trace checkbox toggle."""
        show = self.show_trace_var.get()
        self.ui_state.set_show_trace(show)
        status = "ON" if show else "OFF"
        self.status_msg_var.set(f"Trajectory trace: {status}")
        if self.root:
            self.root.after(1000, lambda: self.status_msg_var.set(""))

    def _toggle_trace(self):
        """Toggle trajectory trace display via keyboard."""
        new_state = self.ui_state.toggle_show_trace()
        self.show_trace_var.set(new_state)
        status = "ON" if new_state else "OFF"
        self.status_msg_var.set(f"Trajectory trace: {status}")
        if self.root:
            self.root.after(1000, lambda: self.status_msg_var.set(""))

    def _increase_points(self):
        """Increase point count."""
        new_count = self.ui_state.increase_points()
        self.points_var.set(str(new_count))
        self.status_msg_var.set(f"Points: {new_count} (press R to apply)")
        if self.root:
            self.root.after(2000, lambda: self.status_msg_var.set(""))

    def _decrease_points(self):
        """Decrease point count."""
        new_count = self.ui_state.decrease_points()
        self.points_var.set(str(new_count))
        self.status_msg_var.set(f"Points: {new_count} (press R to apply)")
        if self.root:
            self.root.after(2000, lambda: self.status_msg_var.set(""))

    def _regenerate_points(self):
        """Trigger regeneration of query points."""
        self.ui_state.should_regenerate_queries = True
        num = self.ui_state.get_num_points()
        self.status_msg_var.set(f"Regenerating {num} points...")
        if self.root:
            self.root.after(2000, lambda: self.status_msg_var.set(""))

    def _on_crop_toggle(self):
        """Handle crop enable/disable toggle."""
        enable = self.enable_crop_var.get()
        self.ui_state.set_enable_crop(enable)
        status = "ON" if enable else "OFF"
        ratio = self.ui_state.get_crop_ratio()
        self.status_msg_var.set(f"Center crop: {status} ({ratio:.0%})")
        if self.root:
            self.root.after(1500, lambda: self.status_msg_var.set(""))

    def _on_crop_ratio_change(self, value):
        """Handle crop ratio slider change."""
        ratio = float(value)
        self.ui_state.set_crop_ratio(ratio)
        self.crop_ratio_label.config(text=f"{ratio:.0%}")

    def _get_realtime_description(self) -> str:
        """Get description text for real-time mode."""
        if self.ui_state.get_realtime_mode():
            return "Latest frame only (skip if slow)"
        else:
            return "Sequential (process all frames)"

    def _on_realtime_toggle(self):
        """Handle real-time mode checkbox toggle."""
        enabled = self.realtime_var.get()
        self.ui_state.set_realtime_mode(enabled)
        self.realtime_desc_var.set(self._get_realtime_description())
        status = "ON" if enabled else "OFF"
        self.status_msg_var.set(f"Real-time mode: {status}")
        if self.root:
            self.root.after(1500, lambda: self.status_msg_var.set(""))

    def _toggle_realtime(self):
        """Toggle real-time mode via keyboard."""
        new_state = self.ui_state.toggle_realtime_mode()
        self.realtime_var.set(new_state)
        self.realtime_desc_var.set(self._get_realtime_description())
        status = "ON" if new_state else "OFF"
        self.status_msg_var.set(f"Real-time mode: {status}")
        if self.root:
            self.root.after(1500, lambda: self.status_msg_var.set(""))

    def _start_point_selection(self):
        """Start point selection mode - pause video and enable clicking."""
        if self.ui_state.get_point_selection_mode():
            return  # Already in selection mode
        self.ui_state.start_point_selection()
        self._update_selection_ui(True)
        self.status_var.set("Status: SELECTING POINTS")
        self.pause_button_text.set("Resume")
        self.status_msg_var.set("Click on video to add points, then click Done")

    def _finish_point_selection(self):
        """Finish selection and add all selected points to tracking."""
        count = self.ui_state.get_selected_point_count()
        if count == 0:
            self.status_msg_var.set("No points selected! Add points or Cancel.")
            if self.root:
                self.root.after(2000, lambda: self.status_msg_var.set(""))
            return

        # Trigger finishing selection
        self.ui_state.trigger_finish_selection()
        self._update_selection_ui(False)
        self.status_var.set("Status: Running")
        self.pause_button_text.set("Pause")
        self.status_msg_var.set(f"Adding {count} new points to tracking...")
        if self.root:
            self.root.after(2000, lambda: self.status_msg_var.set(""))

    def _cancel_point_selection(self):
        """Cancel point selection without applying."""
        self.ui_state.stop_point_selection()
        self.ui_state.set_paused(False)
        self._update_selection_ui(False)
        self.status_var.set("Status: Running")
        self.pause_button_text.set("Pause")
        self.status_msg_var.set("Selection cancelled")
        if self.root:
            self.root.after(1500, lambda: self.status_msg_var.set(""))

    def _cancel_or_quit(self):
        """Cancel selection if in selection mode, otherwise quit."""
        if self.ui_state.get_point_selection_mode():
            self._cancel_point_selection()
        else:
            self._quit()

    def _update_selection_ui(self, in_selection_mode: bool):
        """Update UI elements based on selection mode state."""
        if in_selection_mode:
            self.start_sel_btn.config(state='disabled')
            self.done_sel_btn.config(state='normal')
            self.cancel_sel_btn.config(state='normal')
            self.point_help_var.set("L-click: add | R-click: remove | Enter: done")
        else:
            self.start_sel_btn.config(state='normal')
            self.done_sel_btn.config(state='disabled')
            self.cancel_sel_btn.config(state='disabled')
            self.point_help_var.set("Pause video, click to add points, then Done")
            self.selected_count_var.set("")

    def update_selected_points(self, count: int):
        """Update selected points count (called from pipeline)."""
        if self.root is None:
            return

        def _update():
            if count > 0:
                self.selected_count_var.set(f"Selected: {count} points")
            else:
                self.selected_count_var.set("")

        if self.root:
            self.root.after(0, _update)

    def _on_progress_change(self, value):
        """Handle progress slider change - request seek."""
        target_frame = int(float(value))
        current = self.ui_state.get_current_frame()
        # Only seek if the change is significant (more than 1 frame)
        if abs(target_frame - current) > 1:
            self.ui_state.request_seek(target_frame)
            self.status_msg_var.set(f"Seeking to frame {target_frame}...")

    def _seek_relative(self, delta: int):
        """Seek relative to current position."""
        current = self.ui_state.get_current_frame()
        total = self.ui_state.get_total_frames()
        target = max(0, min(total - 1, current + delta))
        self.ui_state.request_seek(target)
        self.progress_var.set(target)
        self.status_msg_var.set(f"Seeking to frame {target}...")

    def update_progress(self, current_frame: int, total_frames: int = None):
        """Update progress display (called from pipeline)."""
        if self.root is None:
            return

        # Update total frames if provided
        if total_frames is not None and total_frames > 0:
            self.ui_state.set_total_frames(total_frames)

        total = self.ui_state.get_total_frames()
        self.ui_state.set_current_frame(current_frame)

        # Update UI in thread-safe way
        def _update():
            # Show progress frame if we have total frames
            if total > 0 and hasattr(self, 'progress_frame'):
                try:
                    self.progress_frame.grid()
                    self.progress_slider.config(to=total - 1)
                except Exception:
                    pass

            if total > 0:
                self.frame_pos_var.set(f"Frame: {current_frame} / {total}")
                # Only update slider if not being dragged
                if not self.ui_state.has_seek_request():
                    self.progress_var.set(current_frame)
            else:
                self.frame_pos_var.set(f"Frame: {current_frame}")

        if self.root:
            self.root.after(0, _update)

    def _quit(self):
        """Signal quit and close UI."""
        self.ui_state.set_quit()
        if self.root:
            self.root.quit()
            self.root.destroy()

    def start(self):
        """Start the UI in a separate thread."""
        if not self._display_available:
            CONSOLE.print("[yellow][UI][/yellow] Skipping UI start - no display available")
            return
        self.thread = threading.Thread(target=self._create_ui, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the UI."""
        if self.root:
            try:
                self.root.quit()
            except Exception:
                pass

    def is_available(self) -> bool:
        """Check if UI is available."""
        return self._display_available


class InteractiveStreamingPipeline:
    """Real-time streaming tracking pipeline with interactive UI controls."""

    def __init__(
        self,
        config: StreamingConfig,
        enable_ui: bool = True,
        keyframe_config: KeyframeConfig = None
    ):
        """
        Initialize streaming pipeline.

        Args:
            config: StreamingConfig instance
            enable_ui: Whether to enable the interactive UI controls
            keyframe_config: Configuration for automatic keyframe detection
        """
        self.config = config
        self.enable_ui = enable_ui
        self.keyframe_config = keyframe_config or KeyframeConfig()

        # Store frame size for coordinate mapping (set when first frame is processed)
        self._display_frame_size = None  # (H, W)
        self._mouse_callback_installed = False

        # Initialize components
        self.source = StreamingSourceFactory.create(config.source)
        self.query_generator = QueryGeneratorFactory.create(config.query)
        self.tracker = Tracker(config.model)

        # Frame buffer for accumulating frames
        self.frame_buffer = []
        self.all_frames = []  # Store all frames for global tracking

        # Query points (initialized on first frame)
        self.queries = None
        self.query_frame_id = 0

        # Keyframe tracking
        self.keyframes: List[int] = []  # List of keyframe indices
        self.last_keyframe_idx: int = 0  # Index of last keyframe
        self.initial_positions: Optional[np.ndarray] = None  # Positions at last keyframe

        # Tracking results (global)
        self.trajectories = None
        self.visibility = None
        self.dense_tracks = None
        self.dense_visibility = None

        # Intermediate tracking for auto keyframe detection
        self._current_positions: Optional[np.ndarray] = None
        self._current_visibility: Optional[np.ndarray] = None

        # Trajectory history for trace visualization (streaming mode)
        self.tracks_history: deque = deque(maxlen=50)  # Store recent (tracks, visibility) pairs
        self.vis_history: deque = deque(maxlen=50)

        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.processing_times = []

        # Output video writer
        self.video_writer = None

        # UI state and controller
        self.ui_state = UIState()
        self.ui_controller = None

        # Create output directory
        if config.output.output_dir:
            Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)

    def _apply_center_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply center crop to frame based on UI state settings.

        Args:
            frame: Input frame (H, W, 3) numpy array

        Returns:
            Cropped frame if crop is enabled, otherwise original frame
        """
        if not self.ui_state.get_enable_crop():
            return frame

        ratio = self.ui_state.get_crop_ratio()
        if ratio >= 1.0:
            return frame

        H, W = frame.shape[:2]
        new_H = int(H * ratio)
        new_W = int(W * ratio)

        # Calculate crop boundaries (center crop)
        top = (H - new_H) // 2
        left = (W - new_W) // 2

        # Crop the frame
        cropped = frame[top:top + new_H, left:left + new_W]

        # Resize back to original size for consistent processing
        cropped = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

        return cropped

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for OpenCV window.
        Left-click: Add a point at the clicked location
        Right-click: Remove the nearest point within threshold
        """
        if not self.ui_state.get_point_selection_mode():
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add point to selection
            self.ui_state.add_selected_point(float(x), float(y))
            count = self.ui_state.get_selected_point_count()
            CONSOLE.print(f"[cyan]Added point at ({x}, {y}) - {count} selected")
            if self.ui_controller:
                self.ui_controller.update_selected_points(count)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove nearest point from selection
            if self.ui_state.remove_nearby_point(float(x), float(y), threshold=15.0):
                count = self.ui_state.get_selected_point_count()
                CONSOLE.print(f"[cyan]Removed point near ({x}, {y}) - {count} selected")
                if self.ui_controller:
                    self.ui_controller.update_selected_points(count)

    def _setup_mouse_callback(self):
        """Set up mouse callback for the OpenCV window."""
        if not self._mouse_callback_installed and self.config.visualization.show_live:
            cv2.namedWindow(self.config.visualization.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.config.visualization.window_name, self._mouse_callback)
            self._mouse_callback_installed = True

    def _create_queries_from_points(
        self,
        points: List[Tuple[float, float]],
        frame_idx: int,
        device: str
    ) -> Optional[torch.Tensor]:
        """
        Create query tensor from a list of points.

        Args:
            points: List of (x, y) coordinates
            frame_idx: Frame index for query timestamps (use 0 for reinitialization)
            device: Device to create tensor on

        Returns:
            Queries tensor [1, N, 3] with (t, x, y) format, or None if no points
        """
        if not points:
            return None

        # Create queries tensor
        n_points = len(points)
        queries = torch.zeros((1, n_points, 3), dtype=torch.float32, device=device)

        for i, (x, y) in enumerate(points):
            queries[0, i, 0] = frame_idx  # t (should be 0 for reinitialization)
            queries[0, i, 1] = x  # x
            queries[0, i, 2] = y  # y

        return queries

    def _reinitialize_tracking(
        self,
        frame: np.ndarray,
        frame_idx: int,
        device: str,
        queries: Optional[torch.Tensor] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        auto_generate: bool = False
    ) -> bool:
        """
        Reinitialize tracking with new queries or points.
        Used by both regenerate and point selection.

        Args:
            frame: Frame to initialize tracking on
            frame_idx: Actual frame index in video (for keyframes list)
            device: Device to use
            queries: Pre-created queries tensor, OR
            points: List of (x, y) points to create queries from, OR
            auto_generate: If True, auto-generate queries using _initialize_queries

        Returns:
            True if successful, False otherwise
        """
        # Create queries based on input
        # IMPORTANT: Use t=0 for query timestamps because streaming_init resets current_frame_idx to 0
        # The tracker checks active_mask = (query_t <= current_frame_idx), so queries must have t=0
        if queries is None:
            if points is not None:
                # Use t=0 for reinitialization (NOT the actual frame_idx)
                queries = self._create_queries_from_points(points, 0, device)
            elif auto_generate:
                queries = self._initialize_queries(frame, device)

        if queries is None:
            CONSOLE.print("[red]No queries to initialize tracking with")
            return False

        # Reset streaming tracker
        self.tracker.streaming_reset()

        # Update queries and initial positions
        self.queries = queries
        self.initial_positions = self.queries[0, :, 1:3].cpu().numpy()

        # Reinitialize streaming mode
        H, W = frame.shape[:2]
        self.tracker.streaming_init(
            video_shape=(3, H, W),
            initial_queries=self.queries[0]
        )

        # Reset keyframes to current frame (actual video frame index)
        self.keyframes = [frame_idx]

        # Clear trajectory history for fresh start
        self.tracks_history.clear()
        self.vis_history.clear()

        n_points = self.queries.shape[1]
        CONSOLE.print(f"[green]Tracking reinitialized with {n_points} points at frame {frame_idx}")

        return True

    def _draw_selected_points(self, frame: np.ndarray, n_existing: int = 0) -> np.ndarray:
        """
        Draw manually selected points on the frame (in selection mode).
        Points are drawn in the same style as tracked points for consistency.

        Args:
            frame: Input frame (H, W, 3) - will be modified in place
            n_existing: Number of existing tracked points (for color indexing)

        Returns:
            Frame with selected points drawn
        """
        if not self.ui_state.get_point_selection_mode():
            return frame

        # Draw selected points in the same style as tracked points
        selected = self.ui_state.get_selected_points()
        n_selected = len(selected)
        total_points = n_existing + n_selected

        for i, (x, y) in enumerate(selected):
            ix, iy = int(x), int(y)
            if 0 <= ix < frame.shape[1] and 0 <= iy < frame.shape[0]:
                # Use consistent color with tracked points (offset by n_existing)
                color = self._get_point_color(n_existing + i, total_points)
                # Draw filled circle (same as tracked points)
                cv2.circle(frame, (ix, iy), self.config.visualization.point_radius, color, -1)

        # Draw selection mode indicator
        text = f"SELECTION MODE: {n_selected} new points | L-click:add R-click:remove | Enter:done ESC:cancel"
        # Draw background for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (5, frame.shape[0] - 25), (text_size[0] + 15, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def _initialize_queries(
        self,
        first_frame: np.ndarray,
        device: str
    ) -> torch.Tensor:
        """Initialize query points from first frame."""
        # Convert frame to tensor (1, 1, C, H, W)
        frame_tensor = torch.from_numpy(first_frame).to(device)
        frame_tensor = frame_tensor.permute(2, 0, 1).float()
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)

        # Generate queries with initial_frame parameter
        queries = self.query_generator.generate(
            frame_tensor,
            seg_mask=None,
            initial_frame=self.config.query.initial_frame
        )

        # Limit number of points if configured
        max_points = self.config.processing.max_points
        if queries.shape[1] > max_points:
            perm = torch.randperm(queries.shape[1])[:max_points]
            perm = perm.sort().values
            queries = queries[:, perm]

        if self.config.verbose:
            CONSOLE.print(f"[green]Initialized {queries.shape[1]} query points")

        return queries

    def _add_queries_at_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        device: str
    ) -> torch.Tensor:
        """Add new query points at a specific frame (keyframe)."""
        # Convert frame to tensor (1, 1, C, H, W)
        frame_tensor = torch.from_numpy(frame).to(device)
        frame_tensor = frame_tensor.permute(2, 0, 1).float()
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)

        # Generate new queries at this frame
        new_queries = self.query_generator.generate(
            frame_tensor,
            seg_mask=None,
            initial_frame=frame_idx  # Use frame_idx as the query time
        )

        # Limit number of new points
        max_new_points = min(100, self.config.processing.max_points // 5)  # Add up to 100 new points
        if new_queries.shape[1] > max_new_points:
            perm = torch.randperm(new_queries.shape[1])[:max_new_points]
            perm = perm.sort().values
            new_queries = new_queries[:, perm]

        # Update the time index for new queries
        new_queries[:, :, 0] = frame_idx

        if self.config.verbose:
            CONSOLE.print(f"[green]Added {new_queries.shape[1]} new query points at frame {frame_idx}")

        return new_queries

    def _should_create_keyframe_auto(
        self,
        frame_idx: int,
        current_positions: Optional[np.ndarray] = None,
        initial_positions: Optional[np.ndarray] = None,
        visibility: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if current frame should be a keyframe (auto mode).

        This implements similar logic to SLAM's _should_create_keyframe method:
        1. Minimum frame interval since last keyframe
        2. Parallax threshold (median motion from last keyframe)
        3. Visible ratio threshold (too few visible points relative to target)

        Args:
            frame_idx: Current frame index
            current_positions: Current point positions (N, 2)
            initial_positions: Point positions at last keyframe (N, 2)
            visibility: Point visibility mask (N,)

        Returns:
            (should_create, reason): Tuple of boolean and reason string
        """
        cfg = self.keyframe_config

        # Check minimum interval
        frames_since_last = frame_idx - self.last_keyframe_idx
        if frames_since_last < cfg.min_interval:
            return False, f"interval {frames_since_last} < {cfg.min_interval}"

        # If no tracking info available, use interval only
        if current_positions is None or initial_positions is None or visibility is None:
            return True, f"interval {frames_since_last} >= {cfg.min_interval}"

        # Count visible points
        n_visible = int(visibility.sum()) if hasattr(visibility, 'sum') else np.sum(visibility)

        # Use target points from UI state for threshold calculation
        target_points = self.ui_state.get_num_points()
        min_visible_threshold = int(cfg.min_visible_ratio * target_points)

        # Check if too few visible points (force create keyframe)
        # Threshold is min_visible_ratio * target_points (e.g., 0.2 * 64 = 12.8 -> 12)
        if n_visible < min_visible_threshold:
            return True, f"visible {n_visible} < {min_visible_threshold} ({cfg.min_visible_ratio:.0%} of {target_points})"

        # Compute parallax (median motion)
        if n_visible > 0:
            visible_mask = visibility > 0.5 if hasattr(visibility, '__gt__') else visibility.astype(bool)
            if hasattr(visible_mask, 'cpu'):
                visible_mask = visible_mask.cpu().numpy()
            if hasattr(current_positions, 'cpu'):
                curr_pts = current_positions.cpu().numpy()
            else:
                curr_pts = current_positions
            if hasattr(initial_positions, 'cpu'):
                init_pts = initial_positions.cpu().numpy()
            else:
                init_pts = initial_positions

            # Compute motion for visible points
            motion = np.linalg.norm(curr_pts[visible_mask] - init_pts[visible_mask], axis=1)
            median_parallax = np.median(motion) if len(motion) > 0 else 0

            if median_parallax >= cfg.min_parallax:
                return True, f"parallax {median_parallax:.1f} >= {cfg.min_parallax}"

        return False, "no condition met"

    def _update_keyframe_state(self, frame_idx: int, positions: np.ndarray):
        """Update keyframe state after creating a new keyframe."""
        self.last_keyframe_idx = frame_idx
        self.initial_positions = positions.copy() if positions is not None else None

    def _handle_keyframe_in_streaming(
        self,
        frame: np.ndarray,
        frame_idx: int,
        current_tracks: np.ndarray,
        visibility: np.ndarray,
        device: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle keyframe creation in streaming mode.

        This method:
        1. Keeps visible points from current tracking
        2. Discards invisible points
        3. Generates new points to fill up to target count
        4. Re-initializes streaming mode with the combined queries

        Args:
            frame: Current frame (H, W, 3) numpy array
            frame_idx: Current frame index
            current_tracks: Current point positions (N, 2) numpy array
            visibility: Point visibility mask (N,) numpy array or bool array
            device: Device string ('cuda' or 'cpu')

        Returns:
            new_tracks: Updated point positions (N_new, 2)
            new_visibility: Updated visibility mask (N_new,)
        """
        target_points = self.ui_state.get_num_points()

        # Convert visibility to bool mask
        if visibility.dtype == np.bool_:
            vis_mask = visibility
        else:
            vis_mask = visibility > 0.5

        n_visible = int(vis_mask.sum())
        CONSOLE.print(f"[cyan]Keyframe: {n_visible} visible points, target: {target_points}")

        # Keep visible points
        visible_tracks = current_tracks[vis_mask]  # (n_visible, 2)

        # Calculate how many new points we need
        n_new_needed = max(0, target_points - n_visible)

        if n_new_needed > 0:
            # Generate new query points at this frame
            H, W = frame.shape[:2]
            frame_tensor = torch.from_numpy(frame).to(device)
            frame_tensor = frame_tensor.permute(2, 0, 1).float()
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)

            # Generate new queries
            new_queries = self.query_generator.generate(
                frame_tensor,
                seg_mask=None,
                initial_frame=frame_idx
            )
            # new_queries is (1, N, 3) with (t, x, y)
            new_points = new_queries[0, :, 1:3].cpu().numpy()  # (N, 2) x, y

            # Filter out points too close to existing visible points
            if n_visible > 0 and len(new_points) > 0:
                # Compute distances to visible points
                from scipy.spatial.distance import cdist
                distances = cdist(new_points, visible_tracks)  # (N_new, n_visible)
                min_distances = distances.min(axis=1)  # (N_new,)
                # Keep points that are at least 10 pixels away from existing
                far_enough = min_distances > 10.0
                new_points = new_points[far_enough]

            # Limit to needed count
            if len(new_points) > n_new_needed:
                # Random sample
                indices = np.random.choice(len(new_points), n_new_needed, replace=False)
                new_points = new_points[indices]

            CONSOLE.print(f"[cyan]Adding {len(new_points)} new points")

            # Combine visible tracks with new points
            if n_visible > 0:
                combined_tracks = np.vstack([visible_tracks, new_points])
            else:
                combined_tracks = new_points
        else:
            combined_tracks = visible_tracks

        n_combined = len(combined_tracks)
        CONSOLE.print(f"[green]Keyframe created with {n_combined} points ({n_visible} kept + {n_combined - n_visible} new)")

        # Create new queries tensor for streaming init: (N, 3) with (t, x, y)
        new_queries_tensor = torch.zeros(n_combined, 3, device=device)
        new_queries_tensor[:, 0] = frame_idx  # Time index
        new_queries_tensor[:, 1:3] = torch.from_numpy(combined_tracks).to(device)  # x, y

        # Reset streaming mode before re-initializing (important!)
        self.tracker.streaming_reset()

        # Re-initialize streaming mode with new queries
        H, W = frame.shape[:2]
        self.tracker.streaming_init(
            video_shape=(3, H, W),
            initial_queries=new_queries_tensor
        )

        # Process current frame immediately to get valid tracking state
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).to(device)
        result = self.tracker.streaming_process_frame(frame_tensor)
        new_tracks = result['tracks'].cpu().numpy()  # (N, 2)
        new_visibility = result['visibility'].cpu().numpy()  # (N,)

        # Update internal queries (1, N, 3)
        self.queries = new_queries_tensor.unsqueeze(0)

        # Update keyframe state with the actual tracked positions
        self._update_keyframe_state(frame_idx, new_tracks)

        # Clear trajectory history since we changed the point set
        self.tracks_history.clear()
        self.vis_history.clear()

        return new_tracks, new_visibility

    def _process_accumulated_frames(self, device: str) -> Optional[tuple]:
        """Process all accumulated frames with global tracking."""
        if len(self.all_frames) == 0:
            CONSOLE.print("[red]No frames to process!")
            return None

        CONSOLE.print(f"[cyan]Stacking {len(self.all_frames)} frames into video tensor...")

        # Stack all frames into video tensor (1, T, C, H, W)
        video_tensor = torch.cat(self.all_frames, dim=1)

        CONSOLE.print(f"[cyan]Moving video tensor to {device}...")

        # Move video tensor to device
        video_tensor = video_tensor.to(device)

        if self.config.verbose:
            CONSOLE.print(f"[cyan]Video tensor shape: {video_tensor.shape}")
            CONSOLE.print(f"[cyan]Video tensor device: {video_tensor.device}, queries device: {self.queries.device}")
            CONSOLE.print(f"[cyan]Queries shape: {self.queries.shape}")
            # Debug: check query coordinate ranges
            q_xy = self.queries[0, :, 1:3].cpu()  # (N, 2) - x,y coords
            CONSOLE.print(f"[cyan]Query coords - x range: [{q_xy[:,0].min():.1f}, {q_xy[:,0].max():.1f}], y range: [{q_xy[:,1].min():.1f}, {q_xy[:,1].max():.1f}]")

        CONSOLE.print(f"[yellow]Starting tracking... (this may take a while for {video_tensor.shape[1]} frames)")

        # Check GPU memory before tracking
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            CONSOLE.print(f"[cyan]GPU Memory before tracking: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        import sys
        sys.stdout.flush()

        # Run tracking with queries from first frame
        try:
            with torch.no_grad():
                CONSOLE.print(f"[yellow]Calling tracker.track()...")
                sys.stdout.flush()
                result = self.tracker.track(
                    video_tensor,
                    self.queries,
                    use_aug=False
                )
            CONSOLE.print(f"[green]Tracking finished!")
        except Exception as e:
            CONSOLE.print(f"[red]Tracking failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Clean up to free memory
        del video_tensor
        self.all_frames.clear()
        torch.cuda.empty_cache()

        # Handle both dense and non-dense returns
        if len(result) == 4:
            trajectories, visibility, dense_tracks, dense_vis = result
            self.dense_tracks = dense_tracks
            self.dense_visibility = dense_vis
        else:
            trajectories, visibility = result

        # Debug: Print tracking result info
        if self.config.verbose:
            CONSOLE.print(f"[cyan]Tracking result - trajectories: {trajectories.shape}, visibility: {visibility.shape}")
            CONSOLE.print(f"[cyan]Trajectories dtype: {trajectories.dtype}, device: {trajectories.device}")
            CONSOLE.print(f"[cyan]Visibility dtype: {visibility.dtype}, device: {visibility.device}")
            # Sample check
            if trajectories.shape[1] > 0 and trajectories.shape[2] > 0:
                sample_traj = trajectories[0, 0, 0].cpu().numpy()
                sample_vis = visibility[0, 0, 0].cpu().item()
                CONSOLE.print(f"[cyan]Sample point 0 at frame 0: pos={sample_traj}, vis={sample_vis}")

        return trajectories, visibility

    def _visualize_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        trajectories: Optional[torch.Tensor],
        visibility: Optional[torch.Tensor],
        metadata: Dict[str, Any],
        fps: float,
        is_keyframe: bool = False
    ) -> np.ndarray:
        """Visualize tracking results on frame."""
        vis_frame = frame.copy()

        # Draw keyframe indicator
        if is_keyframe:
            cv2.putText(
                vis_frame,
                "KEYFRAME",
                (vis_frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            # Draw border
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1]-1, vis_frame.shape[0]-1), (0, 255, 255), 3)

        # Draw pause indicator if paused
        if self.ui_state.get_paused():
            cv2.putText(
                vis_frame,
                "PAUSED",
                (vis_frame.shape[1] // 2 - 50, vis_frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

        # Draw FPS counter
        if self.config.visualization.display_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                vis_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        # Draw frame info
        if self.config.visualization.display_info:
            info_text = f"Frame: {metadata['frame_id']}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Debug: Draw a test point at center to verify drawing works (ALWAYS)
        if frame_idx == 0:
            center_x, center_y = vis_frame.shape[1] // 2, vis_frame.shape[0] // 2
            cv2.circle(vis_frame, (center_x, center_y), 15, (255, 0, 0), -1)  # Red test point
            CONSOLE.print(f"[cyan]Debug: Drew RED test point at ({center_x}, {center_y})")
            CONSOLE.print(f"[cyan]Debug: trajectories is None: {trajectories is None}")

        # Draw tracking points
        if trajectories is not None and visibility is not None and frame_idx < trajectories.shape[1]:
            current_traj = trajectories[0, frame_idx].cpu().numpy()
            current_vis = visibility[0, frame_idx].cpu().numpy()

            num_high_conf = (current_vis > 0.5).sum()
            num_low_conf = (current_vis <= 0.5).sum()

            # Debug: Log first frame details
            if frame_idx == 0:
                frame_h, frame_w = frame.shape[:2]
                CONSOLE.print(f"[cyan]Debug viz: frame size = {frame_w}x{frame_h}")
                CONSOLE.print(f"[cyan]Debug viz: traj range x=[{current_traj[:,0].min():.1f}, {current_traj[:,0].max():.1f}], y=[{current_traj[:,1].min():.1f}, {current_traj[:,1].max():.1f}]")
                CONSOLE.print(f"[cyan]Debug viz: visibility dtype={current_vis.dtype}")
                # Handle both bool and float visibility
                if current_vis.dtype == np.bool_:
                    num_high_conf = current_vis.sum()
                    num_low_conf = (~current_vis).sum()
                CONSOLE.print(f"[cyan]Debug viz: high conf={num_high_conf}, low conf={num_low_conf}")
                # Count how many points are within bounds
                in_bounds = ((current_traj[:,0] >= 0) & (current_traj[:,0] < frame_w) &
                            (current_traj[:,1] >= 0) & (current_traj[:,1] < frame_h))
                CONSOLE.print(f"[cyan]Debug viz: points in bounds = {in_bounds.sum()}/{len(current_traj)}")

            if self.config.visualization.display_info:
                points_text = f"Points: {num_high_conf}H/{num_low_conf}L/{len(current_vis)}T"
                cv2.putText(
                    vis_frame,
                    points_text,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            # Draw trajectories (trace) - only if show_trace is enabled
            show_trace = self.ui_state.get_show_trace()
            if show_trace and self.config.visualization.tracks_leave_trace > 1:
                trace_start = max(0, frame_idx - self.config.visualization.tracks_leave_trace + 1)
                max_motion = self.config.visualization.max_motion_threshold
                vis_is_bool = visibility.dtype == torch.bool

                for i in range(trajectories.shape[2]):
                    points = []
                    for t in range(trace_start, frame_idx + 1):
                        # Handle both bool and float visibility
                        vis_t = visibility[0, t, i]
                        is_vis = bool(vis_t) if vis_is_bool else (float(vis_t) > 0.5)
                        if is_vis:
                            pt = trajectories[0, t, i].cpu().numpy()
                            if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                                continue
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                                points.append((x, y, t))

                    if len(points) > 1:
                        color = self._get_point_color(i, trajectories.shape[2])
                        for j in range(len(points) - 1):
                            p1 = points[j]
                            p2 = points[j + 1]

                            if max_motion > 0:
                                dx = p2[0] - p1[0]
                                dy = p2[1] - p1[1]
                                motion = (dx * dx + dy * dy) ** 0.5
                                if motion > max_motion:
                                    continue

                            cv2.line(
                                vis_frame,
                                (p1[0], p1[1]),
                                (p2[0], p2[1]),
                                color,
                                self.config.visualization.linewidth
                            )

            # Draw current points
            points_drawn = 0
            points_skipped_bounds = 0
            points_skipped_nan = 0
            for i, (point, vis) in enumerate(zip(current_traj, current_vis)):
                if not np.isfinite(point[0]) or not np.isfinite(point[1]):
                    points_skipped_nan += 1
                    continue

                x, y = int(point[0]), int(point[1])
                if not (0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]):
                    points_skipped_bounds += 1
                    continue

                color = self._get_point_color(i, len(current_traj))
                # Handle both bool and float visibility - use simple truthiness check
                is_visible = bool(vis) if current_vis.dtype == np.bool_ else (float(vis) > 0.5)

                if is_visible:
                    cv2.circle(
                        vis_frame,
                        (x, y),
                        self.config.visualization.point_radius,
                        color,
                        -1  # filled circle
                    )
                    points_drawn += 1
                elif self.config.visualization.show_low_confidence:
                    cv2.circle(
                        vis_frame,
                        (x, y),
                        self.config.visualization.point_radius,
                        color,
                        1  # hollow circle
                    )
                    points_drawn += 1

            # Debug: Log points drawn for first few frames
            if frame_idx < 3:
                CONSOLE.print(f"[cyan]Debug viz frame {frame_idx}: drew {points_drawn}, skipped nan={points_skipped_nan}, out_of_bounds={points_skipped_bounds}")

        return vis_frame

    def _get_point_color(self, idx: int, total: int) -> tuple:
        """Get color for point based on index."""
        hue = int(180 * idx / max(total, 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(map(int, color_bgr))

    def _init_video_writer(self, frame_shape: tuple, is_pairs_mode: bool = False):
        """Initialize video writer for recording."""
        if not self.config.visualization.record_output:
            return

        output_path = self.config.visualization.output_path
        if output_path is None:
            output_path = str(
                Path(self.config.output.output_dir) / "streaming_output.mp4"
            )

        fps = self.config.source.target_fps
        self.video_writer = imageio.get_writer(output_path, fps=fps)

        if self.config.verbose:
            CONSOLE.print(f"[yellow]Recording to: {output_path}")

    def _visualize_streaming_frame(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        visibility: np.ndarray,
        frame_idx: int,
        fps: float,
        is_keyframe: bool = False
    ) -> np.ndarray:
        """Visualize tracking results for streaming mode (single frame)."""
        vis_frame = frame.copy()

        # Save to history for trajectory trace
        if tracks is not None:
            self.tracks_history.append(tracks.copy())
            self.vis_history.append(visibility.copy())

        # Draw keyframe indicator
        if is_keyframe:
            cv2.putText(vis_frame, "KEYFRAME", (vis_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1]-1, vis_frame.shape[0]-1), (0, 255, 255), 3)

        # Draw pause indicator
        if self.ui_state.get_paused():
            cv2.putText(vis_frame, "PAUSED", (vis_frame.shape[1] // 2 - 50, vis_frame.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw FPS
        if self.config.visualization.display_fps:
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Draw frame info
        if self.config.visualization.display_info:
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show keyframe mode
            kf_mode = self.ui_state.get_keyframe_mode().upper()
            mode_color = (0, 255, 0) if kf_mode == 'AUTO' else (255, 165, 0)
            cv2.putText(vis_frame, f"KF: {kf_mode}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

            # Show real-time mode
            rt_mode = "RT" if self.ui_state.get_realtime_mode() else "SEQ"
            rt_color = (0, 255, 255) if self.ui_state.get_realtime_mode() else (200, 200, 200)
            cv2.putText(vis_frame, f"Mode: {rt_mode}", (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, rt_color, 1)

        # Draw tracking points
        if tracks is not None and visibility is not None:
            n_points = len(tracks)
            # Handle bool visibility
            if visibility.dtype == np.bool_:
                vis_mask = visibility
            else:
                vis_mask = visibility > 0.5

            n_visible = vis_mask.sum()
            target_points = self.ui_state.get_num_points()
            cv2.putText(vis_frame, f"Points: {n_visible}/{n_points} (target:{target_points})", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(vis_frame, "+/- pts, R regen", (10, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Draw trajectory traces if enabled
            show_trace = self.ui_state.get_show_trace()
            trace_len = self.config.visualization.tracks_leave_trace
            if show_trace and len(self.tracks_history) > 1:
                # Draw traces for each point
                history_len = min(len(self.tracks_history), trace_len)
                for i in range(n_points):
                    color = self._get_point_color(i, n_points)
                    points_to_draw = []

                    # Collect recent positions for this point
                    for t in range(history_len):
                        hist_idx = len(self.tracks_history) - history_len + t
                        if hist_idx >= 0 and hist_idx < len(self.tracks_history):
                            hist_tracks = self.tracks_history[hist_idx]
                            hist_vis = self.vis_history[hist_idx]
                            if i < len(hist_tracks):
                                pt = hist_tracks[i]
                                vis = hist_vis[i]
                                # Check visibility
                                is_vis = bool(vis) if hist_vis.dtype == np.bool_ else (float(vis) > 0.5)
                                if is_vis and np.isfinite(pt[0]) and np.isfinite(pt[1]):
                                    x, y = int(pt[0]), int(pt[1])
                                    if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                                        points_to_draw.append((x, y))

                    # Draw lines connecting the points
                    if len(points_to_draw) > 1:
                        for j in range(len(points_to_draw) - 1):
                            p1, p2 = points_to_draw[j], points_to_draw[j + 1]
                            # Skip if motion is too large
                            dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                            if dist < self.config.visualization.max_motion_threshold:
                                cv2.line(vis_frame, p1, p2, color, self.config.visualization.linewidth)

            # Draw current points
            for i, (pt, vis) in enumerate(zip(tracks, vis_mask)):
                if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                    continue
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                    color = self._get_point_color(i, n_points)
                    if vis:
                        cv2.circle(vis_frame, (x, y), self.config.visualization.point_radius, color, -1)
                    elif self.config.visualization.show_low_confidence:
                        cv2.circle(vis_frame, (x, y), self.config.visualization.point_radius, color, 1)

        # Draw manually selected points
        vis_frame = self._draw_selected_points(vis_frame)

        return vis_frame

    def run_realtime(self):
        """Run TRUE real-time streaming tracking (process each frame immediately)."""
        display_available = has_display()
        if not display_available:
            CONSOLE.print("[yellow]No display detected - running in headless mode")
            self.config.visualization.show_live = False
            self.enable_ui = False

        if self.config.verbose:
            CONSOLE.print("\n[bold cyan]" + "="*60)
            CONSOLE.print("[bold cyan]Real-time Streaming Tracking Pipeline")
            CONSOLE.print("[bold cyan]" + "="*60 + "\n")
            CONSOLE.print(f"[yellow]Source: {self.config.source.source_type}")
            CONSOLE.print(f"[yellow]Device: {self.config.model.device}")
            CONSOLE.print("[yellow]Mode: Real-time (process latest frame)\n")

        # Get total frames from source (for video files)
        total_frames = 0
        supports_seek = False
        if hasattr(self.source, 'get_total_frames'):
            total_frames = self.source.get_total_frames()
            self.ui_state.set_total_frames(total_frames)
        if hasattr(self.source, 'supports_seek'):
            supports_seek = self.source.supports_seek()

        if total_frames > 0:
            CONSOLE.print(f"[cyan]Video has {total_frames} frames (seeking {'enabled' if supports_seek else 'disabled'})")

        # Start UI controller
        if self.enable_ui and display_available:
            self.ui_controller = StreamingControlUI(self.ui_state)
            self.ui_controller.start()
            time.sleep(0.5)
            # Update UI with total frames
            if total_frames > 0:
                self.ui_controller.update_progress(0, total_frames)

        device = self.config.model.device
        frame_count = 0
        frames_skipped = 0
        start_time = time.time()
        last_process_time = 0
        streaming_initialized = False

        # FPS tracking
        fps_counter = deque(maxlen=30)
        current_fps = 0.0

        # Frame timing for real-time mode
        target_frame_time = 1.0 / self.config.source.target_fps
        last_frame_time = time.time()

        realtime_mode = self.ui_state.get_realtime_mode()
        CONSOLE.print(f"[bold green]Starting tracking... (Real-time mode: {'ON' if realtime_mode else 'OFF'})")

        # Create source iterator
        source_iter = iter(self.source)

        try:
            while True:
                loop_start = time.time()

                # Handle seek request
                seek_frame = self.ui_state.consume_seek_request()
                if seek_frame >= 0 and supports_seek:
                    CONSOLE.print(f"[cyan]Seeking to frame {seek_frame}...")
                    if self.source.seek(seek_frame):
                        frame_count = seek_frame
                        # Don't reset streaming state - continue tracking from new position
                        # The tracker will continue using its memory to track points
                        # Only clear trajectory display history (visual only)
                        self.tracks_history.clear()
                        self.vis_history.clear()
                        # The source will handle the seek on next read

                # Handle pause BEFORE reading new frame (to actually pause video input)
                # In non-realtime mode, we want to freeze on the current frame
                if self.ui_state.get_paused() and not self.ui_state.get_realtime_mode():
                    if self.config.visualization.show_live:
                        # Use the last frame we had (or current frame if none stored)
                        if hasattr(self, '_last_frame') and self._last_frame is not None:
                            pause_frame = self._last_frame.copy()
                        else:
                            # No last frame yet, need to read one
                            try:
                                frame, metadata = next(source_iter)
                                if 'frame_id' in metadata:
                                    frame_count = metadata['frame_id']
                                frame = self._apply_center_crop(frame)
                                self._last_frame = frame.copy()
                                pause_frame = frame.copy()
                            except StopIteration:
                                break

                        # If in point selection mode, draw existing tracked points and selected points
                        if self.ui_state.get_point_selection_mode() and streaming_initialized:
                            # Draw existing tracked points (from last tracking result)
                            n_tracked = 0
                            if hasattr(self, '_last_tracks') and self._last_tracks is not None:
                                n_tracked = len(self._last_tracks)
                                for i, (pt, vis) in enumerate(zip(self._last_tracks, self._last_visibility)):
                                    if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                                        continue
                                    is_vis = bool(vis) if self._last_visibility.dtype == np.bool_ else (float(vis) > 0.5)
                                    x, y = int(pt[0]), int(pt[1])
                                    if 0 <= x < pause_frame.shape[1] and 0 <= y < pause_frame.shape[0]:
                                        color = self._get_point_color(i, n_tracked)
                                        if is_vis:
                                            cv2.circle(pause_frame, (x, y), self.config.visualization.point_radius, color, -1)
                                        else:
                                            cv2.circle(pause_frame, (x, y), self.config.visualization.point_radius, color, 1)

                            # Draw selected points (with consistent color indexing)
                            pause_frame = self._draw_selected_points(pause_frame, n_tracked)
                        else:
                            # Normal pause - just show paused message
                            cv2.putText(pause_frame, "PAUSED - Space to Resume",
                                       (pause_frame.shape[1]//2 - 150, pause_frame.shape[0]//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.imshow(self.config.visualization.window_name,
                                  cv2.cvtColor(pause_frame, cv2.COLOR_RGB2BGR))
                        key = cv2.waitKey(100) & 0xFF
                        self._handle_key(key, self._last_frame, frame_count, device)

                        # Update UI with selected point count
                        if self.ui_state.get_point_selection_mode() and self.ui_controller:
                            self.ui_controller.update_selected_points(self.ui_state.get_selected_point_count())

                    # Check for finish selection request (Done button clicked)
                    if self.ui_state.consume_finish_selection() and streaming_initialized:
                        selected_points = self.ui_state.get_selected_points()
                        n_selected = len(selected_points)

                        if n_selected > 0:
                            CONSOLE.print(f"[cyan]Adding {n_selected} new points to tracking...")

                            # Get all current tracked points
                            if hasattr(self, '_last_tracks') and self._last_tracks is not None:
                                current_tracks = self._last_tracks
                                n_current = len(current_tracks)
                            else:
                                current_tracks = np.array([])
                                n_current = 0

                            # Combine: current tracked points + new selected points
                            all_points = []
                            for pt in current_tracks:
                                all_points.append((float(pt[0]), float(pt[1])))
                            for x, y in selected_points:
                                all_points.append((x, y))

                            n_total = len(all_points)

                            # Reinitialize tracking with combined points
                            # Use _last_frame as the reference frame for initialization
                            success = self._reinitialize_tracking(
                                frame=self._last_frame,
                                frame_idx=frame_count,
                                device=device,
                                points=all_points
                            )

                            if success:
                                self.ui_state.set_num_points(n_total)
                                CONSOLE.print(f"[green]Now tracking {n_total} points ({n_current} existing + {n_selected} new)")

                        # Exit selection mode and resume
                        self.ui_state.stop_point_selection()
                        self.ui_state.set_paused(False)

                    continue

                # Get frame (with optional skipping in real-time mode)
                try:
                    frame, metadata = next(source_iter)
                except StopIteration:
                    break

                # Update current frame in UI state
                if 'frame_id' in metadata:
                    frame_count = metadata['frame_id']
                self.ui_state.set_current_frame(frame_count)

                # Update UI progress
                if self.ui_controller is not None and total_frames > 0:
                    self.ui_controller.update_progress(frame_count, total_frames)

                # In real-time mode, skip frames if processing is too slow
                realtime_mode = self.ui_state.get_realtime_mode()
                if realtime_mode and streaming_initialized:
                    elapsed_since_last = time.time() - last_frame_time
                    # If we're behind, skip frames to catch up
                    if elapsed_since_last > target_frame_time * 2:
                        frames_to_skip = int(elapsed_since_last / target_frame_time) - 1
                        for _ in range(min(frames_to_skip, 10)):  # Skip up to 10 frames
                            try:
                                frame, metadata = next(source_iter)
                                frames_skipped += 1
                                if 'frame_id' in metadata:
                                    frame_count = metadata['frame_id']
                            except StopIteration:
                                break

                last_frame_time = time.time()

                # Apply center crop if enabled
                frame = self._apply_center_crop(frame)

                # Check for quit
                if self.ui_state.get_quit():
                    CONSOLE.print("\n[yellow]User requested quit")
                    break

                # Initialize on first frame
                if not streaming_initialized:
                    CONSOLE.print("[cyan]Initializing tracker...")

                    # Use num_points from UI state
                    num_points = self.ui_state.get_num_points()
                    self.config.processing.max_points = num_points

                    # Generate initial queries
                    self.queries = self._initialize_queries(frame, device)
                    n_queries = self.queries.shape[1]
                    self.initial_positions = self.queries[0, :, 1:3].cpu().numpy()
                    CONSOLE.print(f"[green]Generated {n_queries} query points")

                    # Initialize streaming mode
                    H, W = frame.shape[:2]
                    self.tracker.streaming_init(
                        video_shape=(3, H, W),
                        initial_queries=self.queries[0]  # [N, 3] tensor
                    )
                    streaming_initialized = True
                    self.keyframes.append(0)
                    self._init_video_writer(frame.shape)
                    self._setup_mouse_callback()  # Set up mouse callback for point selection
                    CONSOLE.print("[green]Streaming mode initialized!")
                    CONSOLE.print("[cyan]Controls: +/- pts, R regen, P select points, Space pause, Q quit")

                # Convert frame to tensor
                frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # [C, H, W]
                frame_tensor = frame_tensor.to(device)

                # Process frame
                process_start = time.time()
                result = self.tracker.streaming_process_frame(frame_tensor)
                process_time = time.time() - process_start

                tracks = result['tracks'].cpu().numpy()  # [N, 2]
                visibility = result['visibility'].cpu().numpy()  # [N]

                # Check for regenerate queries request (dynamic point count adjustment)
                if self.ui_state.consume_regenerate_queries():
                    num_points = self.ui_state.get_num_points()
                    self.config.processing.max_points = num_points
                    CONSOLE.print(f"[cyan]Regenerating {num_points} query points...")

                    # Reinitialize tracking with auto-generated queries
                    self._reinitialize_tracking(
                        frame=frame,
                        frame_idx=frame_count,
                        device=device,
                        auto_generate=True
                    )

                # Check for keyframe request (manual)
                manual_keyframe = self.ui_state.consume_add_keyframe()

                # Auto keyframe check
                auto_keyframe = False
                auto_reason = ""
                if self.ui_state.is_auto_keyframe() and frame_count > 0:
                    auto_keyframe, auto_reason = self._should_create_keyframe_auto(
                        frame_count, tracks, self.initial_positions, visibility
                    )

                # Handle keyframe creation (manual or auto)
                should_create_kf = (manual_keyframe or auto_keyframe) and frame_count not in self.keyframes
                if should_create_kf:
                    self.keyframes.append(frame_count)
                    mode_str = "Manual" if manual_keyframe else f"Auto ({auto_reason})"
                    CONSOLE.print(f"[green]Keyframe at frame {frame_count} [{mode_str}]")

                    # Handle keyframe: keep visible, discard invisible, add new points
                    tracks, visibility = self._handle_keyframe_in_streaming(
                        frame, frame_count, tracks, visibility, device
                    )

                # Store last tracks/visibility for use in paused mode (point selection)
                self._last_tracks = tracks.copy()
                self._last_visibility = visibility.copy()
                self._last_frame = frame.copy()

                # Visualize
                is_keyframe = frame_count in self.keyframes
                vis_frame = self._visualize_streaming_frame(
                    frame, tracks, visibility, frame_count, current_fps, is_keyframe
                )

                # Display
                if self.config.visualization.show_live:
                    cv2.imshow(self.config.visualization.window_name,
                              cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key, frame, frame_count, device)

                # Record
                if self.video_writer is not None:
                    self.video_writer.append_data(vis_frame)

                # Update FPS
                loop_time = time.time() - loop_start
                fps_counter.append(loop_time)
                if len(fps_counter) > 0:
                    current_fps = 1.0 / (sum(fps_counter) / len(fps_counter))

                frame_count += 1

                # Progress output
                if self.config.verbose and frame_count % 30 == 0:
                    n_vis = (visibility > 0.5).sum() if visibility.dtype != np.bool_ else visibility.sum()
                    CONSOLE.print(f"[cyan]Frame {frame_count} | FPS: {current_fps:.1f} | "
                                 f"Visible: {n_vis}/{len(visibility)} | "
                                 f"Process: {process_time*1000:.1f}ms")

                # Check max duration
                if self.config.max_duration and (time.time() - start_time) > self.config.max_duration:
                    CONSOLE.print("\n[yellow]Max duration reached")
                    break

        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Interrupted by user")
        except Exception as e:
            CONSOLE.print(f"[red]Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            self.tracker.streaming_reset()
            self._cleanup()

            # Summary
            total_time = time.time() - start_time
            if self.config.verbose and frame_count > 0:
                CONSOLE.print("\n[bold cyan]" + "="*60)
                CONSOLE.print("[bold green]Streaming completed")
                CONSOLE.print(f"Total frames: {frame_count}")
                CONSOLE.print(f"Total time: {total_time:.2f}s")
                CONSOLE.print(f"Average FPS: {frame_count/total_time:.1f}")
                CONSOLE.print("[bold cyan]" + "="*60 + "\n")

    def _handle_key(self, key: int, frame: np.ndarray, frame_count: int, device: str):
        """Handle keyboard input."""
        if key == ord(' '):
            # Space: toggle pause (but not if in selection mode - use Enter/ESC there)
            if not self.ui_state.get_point_selection_mode():
                paused = self.ui_state.get_paused()
                self.ui_state.set_paused(not paused)
        elif key == ord('k') or key == ord('K'):
            self.ui_state.trigger_add_keyframe()
        elif key == ord('m') or key == ord('M'):
            mode = 'manual' if self.ui_state.get_keyframe_mode() == 'auto' else 'auto'
            self.ui_state.set_keyframe_mode(mode)
            CONSOLE.print(f"[cyan]Keyframe mode: {mode.upper()}")
        elif key == ord('t') or key == ord('T'):
            new_state = self.ui_state.toggle_show_trace()
            CONSOLE.print(f"[cyan]Trace: {'ON' if new_state else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            new_count = self.ui_state.increase_points()
            CONSOLE.print(f"[cyan]Points: {new_count} (will regenerate on next keyframe)")
        elif key == ord('-') or key == ord('_'):
            new_count = self.ui_state.decrease_points()
            CONSOLE.print(f"[cyan]Points: {new_count} (will regenerate on next keyframe)")
        elif key == ord('r') or key == ord('R'):
            # Force regenerate queries now
            self.ui_state.should_regenerate_queries = True
            CONSOLE.print(f"[cyan]Regenerating {self.ui_state.get_num_points()} query points...")
        elif key == ord('p') or key == ord('P'):
            # Start point selection mode (if not already in it)
            if not self.ui_state.get_point_selection_mode():
                self.ui_state.start_point_selection()
                CONSOLE.print("[cyan]Selection mode - click to add points, Enter to apply, ESC to cancel")
        elif key == ord('l') or key == ord('L'):
            # Toggle real-time mode
            new_state = self.ui_state.toggle_realtime_mode()
            CONSOLE.print(f"[cyan]Real-time mode: {'ON' if new_state else 'OFF'}")
        elif key == ord('q'):
            self.ui_state.set_quit()
        elif key == 27:  # ESC
            # Cancel selection if in selection mode, otherwise quit
            if self.ui_state.get_point_selection_mode():
                self.ui_state.stop_point_selection()
                self.ui_state.set_paused(False)
                CONSOLE.print("[cyan]Selection cancelled, resuming...")
            else:
                self.ui_state.set_quit()
        elif key == 13 or key == 10:  # Enter
            # Finish selection if in selection mode
            if self.ui_state.get_point_selection_mode():
                count = self.ui_state.get_selected_point_count()
                if count > 0:
                    self.ui_state.trigger_finish_selection()
                    CONSOLE.print(f"[cyan]Applying {count} selected points...")
                else:
                    CONSOLE.print("[yellow]No points selected - add points or press ESC to cancel")

    def run(self):
        """Run streaming tracking pipeline with interactive controls."""
        # Check display availability for headless operation
        display_available = has_display()
        if not display_available:
            CONSOLE.print("[yellow]No display detected - running in headless mode")
            CONSOLE.print("[yellow]Live preview and UI disabled, recording only\n")
            self.config.visualization.show_live = False
            self.enable_ui = False

        if self.config.verbose:
            CONSOLE.print("\n[bold cyan]" + "="*60)
            CONSOLE.print("[bold cyan]Starting Interactive Streaming Tracking Pipeline")
            CONSOLE.print("[bold cyan]" + "="*60 + "\n")
            CONSOLE.print(f"[yellow]Source: {self.config.source.source_type}")
            CONSOLE.print(f"[yellow]Display: {'Available' if display_available else 'Headless mode'}")
            CONSOLE.print(f"[yellow]UI Controls: {'Enabled' if self.enable_ui else 'Disabled'}")
            CONSOLE.print(f"[yellow]Device: {self.config.model.device}\n")

            if self.enable_ui:
                CONSOLE.print("[cyan]UI Controls:")
                CONSOLE.print("  - Space: Pause/Resume")
                CONSOLE.print("  - K: Add keyframe at current position")
                CONSOLE.print("  - M: Toggle Auto/Manual keyframe mode")
                CONSOLE.print("  - T: Toggle trajectory trace display")
                CONSOLE.print("  - Q/ESC: Quit\n")
                CONSOLE.print(f"[yellow]Keyframe Mode: {self.ui_state.get_keyframe_mode().upper()}")
                CONSOLE.print(f"[yellow]Trajectory Trace: {'ON' if self.ui_state.get_show_trace() else 'OFF'}")

        # Start UI controller in separate thread if enabled
        if self.enable_ui and display_available:
            self.ui_controller = StreamingControlUI(self.ui_state)
            self.ui_controller.start()
            time.sleep(0.5)  # Give UI time to start

        device = self.config.model.device
        frame_count = 0
        start_time = time.time()

        # Phase 1: Collect all frames (with pause/keyframe support)
        CONSOLE.print("[bold green]Phase 1: Collecting frames...")
        try:
            for frame, metadata in self.source:
                # Apply center crop if enabled
                frame = self._apply_center_crop(frame)

                # Check for quit
                if self.ui_state.get_quit():
                    CONSOLE.print("\n[yellow]User requested quit")
                    break

                # Handle pause
                while self.ui_state.get_paused() and not self.ui_state.get_quit():
                    # Check for keyframe request while paused
                    if self.ui_state.consume_add_keyframe():
                        self.keyframes.append(frame_count)
                        CONSOLE.print(f"[green]Keyframe added at frame {frame_count}")

                    # Show paused frame
                    if self.config.visualization.show_live:
                        pause_frame = frame.copy()
                        cv2.putText(
                            pause_frame,
                            "PAUSED - Press Space to Resume",
                            (pause_frame.shape[1] // 2 - 180, pause_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )
                        cv2.imshow(
                            self.config.visualization.window_name,
                            cv2.cvtColor(pause_frame, cv2.COLOR_RGB2BGR)
                        )
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord(' '):
                            self.ui_state.set_paused(False)
                        elif key == ord('k') or key == ord('K'):
                            self.keyframes.append(frame_count)
                            CONSOLE.print(f"[green]Keyframe added at frame {frame_count}")
                        elif key == ord('q') or key == 27:
                            self.ui_state.set_quit()
                            break
                    else:
                        time.sleep(0.1)

                if self.ui_state.get_quit():
                    break

                frame_start = time.time()

                # Initialize queries on first frame
                if self.queries is None:
                    self.queries = self._initialize_queries(frame, device)
                    self.query_frame_id = metadata['frame_id']
                    self.keyframes.append(0)  # First frame is always a keyframe
                    self.last_keyframe_idx = 0
                    # Store initial positions for auto keyframe detection
                    # Use grid of query positions as initial
                    self.initial_positions = self.queries[0, :, 1:3].cpu().numpy()  # (N, 2) x,y
                    is_pairs_mode = self.config.visualization.plot_mode == 'pairs'
                    self._init_video_writer(frame.shape, is_pairs_mode=is_pairs_mode)

                # Check for manual keyframe request
                manual_keyframe_requested = self.ui_state.consume_add_keyframe()

                # Check for auto keyframe (only in auto mode)
                auto_keyframe_triggered = False
                auto_reason = ""
                if self.ui_state.is_auto_keyframe() and frame_count > 0:
                    # For auto detection, we use a simple heuristic based on frame interval
                    # since we don't have real-time tracking results during collection
                    auto_keyframe_triggered, auto_reason = self._should_create_keyframe_auto(
                        frame_count,
                        current_positions=None,  # No tracking during collection
                        initial_positions=self.initial_positions,
                        visibility=None
                    )

                # Create keyframe if requested (manual or auto)
                should_create_keyframe = manual_keyframe_requested or auto_keyframe_triggered
                if should_create_keyframe and frame_count not in self.keyframes:
                    self.keyframes.append(frame_count)
                    mode_str = "Manual" if manual_keyframe_requested else f"Auto ({auto_reason})"
                    CONSOLE.print(f"[green]Keyframe added at frame {frame_count} [{mode_str}]")

                    # Add new query points at this keyframe
                    new_queries = self._add_queries_at_frame(frame, frame_count, device)
                    self.queries = torch.cat([self.queries, new_queries], dim=1)

                    # Update keyframe state
                    self._update_keyframe_state(frame_count, self.queries[0, :, 1:3].cpu().numpy())

                # Store frame (make a copy to avoid issues with frame reuse)
                self.frame_buffer.append(frame.copy())

                # Convert to tensor but keep on CPU
                frame_tensor = torch.from_numpy(frame).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
                self.all_frames.append(frame_tensor)

                frame_count += 1

                # Show live preview during collection
                if self.config.visualization.show_live:
                    preview_frame = frame.copy()
                    is_keyframe = frame_count - 1 in self.keyframes

                    if is_keyframe:
                        cv2.putText(
                            preview_frame,
                            "KEYFRAME",
                            (preview_frame.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                        cv2.rectangle(preview_frame, (0, 0),
                                    (preview_frame.shape[1]-1, preview_frame.shape[0]-1),
                                    (0, 255, 255), 3)

                    cv2.putText(
                        preview_frame,
                        f"Collecting: Frame {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        preview_frame,
                        f"Keyframes: {len(self.keyframes)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    # Show keyframe mode
                    kf_mode = self.ui_state.get_keyframe_mode().upper()
                    mode_color = (0, 255, 0) if kf_mode == 'AUTO' else (255, 165, 0)  # Green for auto, orange for manual
                    cv2.putText(
                        preview_frame,
                        f"KF Mode: {kf_mode} (M)",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        mode_color,
                        1
                    )

                    # Show trace status
                    trace_on = self.ui_state.get_show_trace()
                    trace_status = "ON" if trace_on else "OFF"
                    trace_color = (0, 255, 0) if trace_on else (128, 128, 128)  # Green if on, gray if off
                    cv2.putText(
                        preview_frame,
                        f"Trace: {trace_status} (T)",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        trace_color,
                        1
                    )

                    cv2.imshow(
                        self.config.visualization.window_name,
                        cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
                    )

                    # Handle keyboard input during collection
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        self.ui_state.set_paused(True)
                    elif key == ord('k') or key == ord('K'):
                        if frame_count - 1 not in self.keyframes:
                            self.keyframes.append(frame_count - 1)
                            CONSOLE.print(f"[green]Keyframe added at frame {frame_count - 1} [Manual]")
                            new_queries = self._add_queries_at_frame(
                                self.frame_buffer[-1], frame_count - 1, device
                            )
                            self.queries = torch.cat([self.queries, new_queries], dim=1)
                            self._update_keyframe_state(frame_count - 1, self.queries[0, :, 1:3].cpu().numpy())
                    elif key == ord('m') or key == ord('M'):
                        # Toggle keyframe mode
                        current_mode = self.ui_state.get_keyframe_mode()
                        new_mode = 'manual' if current_mode == 'auto' else 'auto'
                        self.ui_state.set_keyframe_mode(new_mode)
                        CONSOLE.print(f"[cyan]Keyframe mode: {new_mode.upper()}")
                    elif key == ord('t') or key == ord('T'):
                        # Toggle trajectory trace display
                        new_state = self.ui_state.toggle_show_trace()
                        status = "ON" if new_state else "OFF"
                        CONSOLE.print(f"[cyan]Trajectory trace: {status}")
                    elif key == ord('q') or key == 27:
                        self.ui_state.set_quit()
                        break

                # Print progress
                if self.config.verbose and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    CONSOLE.print(f"[cyan]Collected {frame_count} frames | Keyframes: {len(self.keyframes)} | Elapsed: {elapsed:.1f}s")

                # Check max duration
                if self.config.max_duration is not None:
                    if time.time() - start_time > self.config.max_duration:
                        CONSOLE.print("\n[yellow]Max duration reached")
                        break

        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Interrupted by user")

        if frame_count == 0:
            CONSOLE.print("[red]No frames collected!")
            self._cleanup()
            return

        # Phase 2: Run tracking on all frames
        CONSOLE.print(f"\n[bold green]Phase 2: Running tracking on {frame_count} frames...")
        CONSOLE.print(f"[yellow]Total queries: {self.queries.shape[1]}")
        CONSOLE.print(f"[yellow]Keyframes: {self.keyframes}")
        CONSOLE.print(f"[yellow]all_frames count: {len(self.all_frames)}")
        CONSOLE.print(f"[yellow]frame_buffer count: {len(self.frame_buffer)}")

        tracking_start = time.time()

        result = self._process_accumulated_frames(device)
        if result is not None:
            self.trajectories, self.visibility = result
            tracking_time = time.time() - tracking_start

            CONSOLE.print(f"[green]Tracking completed in {tracking_time:.2f}s")
            CONSOLE.print(f"[green]Trajectories shape: {self.trajectories.shape}")
            CONSOLE.print(f"[green]Visibility shape: {self.visibility.shape}, dtype: {self.visibility.dtype}")
            CONSOLE.print(f"[green]FPS: {frame_count/tracking_time:.1f}")
        else:
            CONSOLE.print("[red]ERROR: Tracking returned None! Continuing with visualization only...")
            # Don't return - continue to Phase 3 to at least show frames

        # Phase 3: Visualize and save
        CONSOLE.print(f"\n[bold green]Phase 3: Generating visualization...")
        vis_start = time.time()

        # Debug: Check tracking results
        if self.trajectories is not None:
            CONSOLE.print(f"[cyan]Debug: trajectories shape = {self.trajectories.shape}")
            CONSOLE.print(f"[cyan]Debug: visibility shape = {self.visibility.shape}, dtype = {self.visibility.dtype}")
            # Check visibility stats for first frame
            vis_first = self.visibility[0, 0].cpu().numpy()
            # Visibility can be bool or float, handle both
            if vis_first.dtype == np.bool_:
                n_visible = vis_first.sum()
                CONSOLE.print(f"[cyan]Debug: Frame 0 visibility (bool) - visible: {n_visible}/{len(vis_first)}")
            else:
                CONSOLE.print(f"[cyan]Debug: Frame 0 visibility - min={vis_first.min():.3f}, max={vis_first.max():.3f}, mean={vis_first.mean():.3f}")
                CONSOLE.print(f"[cyan]Debug: Frame 0 visible points (>0.5): {(vis_first > 0.5).sum()}/{len(vis_first)}")
            # Check trajectory range
            traj_first = self.trajectories[0, 0].cpu().numpy()
            CONSOLE.print(f"[cyan]Debug: Frame 0 traj - x range=[{traj_first[:,0].min():.1f}, {traj_first[:,0].max():.1f}], y range=[{traj_first[:,1].min():.1f}, {traj_first[:,1].max():.1f}]")
        else:
            CONSOLE.print("[red]Debug: trajectories is None!")

        for idx, frame in enumerate(self.frame_buffer):
            if self.ui_state.get_quit():
                break

            avg_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            is_keyframe = idx in self.keyframes

            vis_frame = self._visualize_frame(
                frame,
                idx,
                self.trajectories,
                self.visibility,
                {'frame_id': idx},
                avg_fps,
                is_keyframe=is_keyframe
            )

            # Show live window
            if self.config.visualization.show_live:
                cv2.imshow(
                    self.config.visualization.window_name,
                    cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    CONSOLE.print("\n[yellow]User requested quit")
                    break

            # Record to video
            if self.video_writer is not None:
                self.video_writer.append_data(vis_frame)

            if self.config.verbose and (idx + 1) % 30 == 0:
                CONSOLE.print(f"[cyan]Visualized {idx + 1}/{frame_count} frames")

        vis_time = time.time() - vis_start
        CONSOLE.print(f"[green]Visualization completed in {vis_time:.2f}s")

        # Cleanup
        self._cleanup()

        # Print summary
        total_time = time.time() - start_time
        if self.config.verbose:
            CONSOLE.print("\n[bold cyan]" + "="*60)
            CONSOLE.print("[bold green]Streaming completed")
            CONSOLE.print("[bold cyan]" + "="*60)
            CONSOLE.print(f"Total frames: {frame_count}")
            CONSOLE.print(f"Total keyframes: {len(self.keyframes)}")
            CONSOLE.print(f"Total queries: {self.queries.shape[1] if self.queries is not None else 0}")
            CONSOLE.print(f"Total time: {total_time:.2f}s")
            CONSOLE.print(f"Overall FPS: {frame_count/total_time:.1f}")
            CONSOLE.print("[bold cyan]" + "="*60 + "\n")

    def _cleanup(self):
        """Clean up resources."""
        # Stop UI
        if self.ui_controller:
            self.ui_controller.stop()

        # Release source
        self.source.release()

        # Close video writer
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Close windows
        if self.config.visualization.show_live:
            cv2.destroyAllWindows()
