from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, Protocol, TypeAlias

import numpy as np

BACKGROUND = "#20252b"
PANEL_BG = "#2a3038"
GRID_MINOR = "#303740"
GRID_MAJOR = "#46515d"
AXIS_X = "#d16d6d"
AXIS_Y = "#89dd8f"
AXIS_Z = "#6da7d1"
TEXT_PRIMARY = "#edf2f7"
TEXT_MUTED = "#9da9b5"
SELECTION = "#f2c14e"
SKY_TOP = "#313a45"
SKY_BOTTOM = "#171b20"
GROUND_FILL = "#1a1f25"
CLOTH_LINE = "#7dcfff"
CLOTH_POINT = "#c9f9ff"
PINNED_POINT = "#ffcb6b"
ScreenPoint: TypeAlias = tuple[float, float]


class SupportsFieldToNumpy(Protocol):
    def to_numpy(self) -> np.ndarray: ...


class SupportsViewerParticles(Protocol):
    inv_mass: SupportsFieldToNumpy

    def positions_numpy(self) -> np.ndarray: ...
    def velocities_numpy(self) -> np.ndarray: ...


class SupportsViewerScene(Protocol):
    def step(self, dt: float) -> None: ...


SceneFactory = Callable[[], tuple[SupportsViewerScene, SupportsViewerParticles]]


@dataclass(slots=True)
class PickInfo:
    kind: str
    index: int
    name: str
    note: str


class DCCViewer(tk.Tk):
    def __init__(
        self,
        scene_factory: SceneFactory | None = None,
        title: str = "Dynames Viewer",
        scene_name: str = "Scene",
    ) -> None:
        super().__init__()
        self.title(title)
        self.geometry("1480x900")
        self.minsize(1180, 720)
        self.configure(bg=BACKGROUND)

        self.scene_name = scene_name
        self.scene: SupportsViewerScene | None = None
        self.particles: SupportsViewerParticles | None = None
        self.dt = 1.0 / 60.0
        self.frame_index = 0
        self.is_playing = scene_factory is not None
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.velocities = np.zeros((0, 3), dtype=np.float32)
        self.inv_mass = np.zeros(0, dtype=np.float32)
        self.grid_width = 0
        self.grid_height = 0

        if scene_factory is not None:
            self.scene, self.particles = scene_factory()
            self.positions = self.particles.positions_numpy()
            self.velocities = self.particles.velocities_numpy()
            self.inv_mass = self.particles.inv_mass.to_numpy()
            self.grid_width, self.grid_height = self._infer_grid_shape(
                self.positions.shape[0]
            )

        default_note = (
            "Live simulation viewport driven by the selected pipeline."
            if scene_factory is not None
            else "Generic viewport tool. Pick a pipeline with --pipeline and add --viewer to inspect a running simulation."
        )
        self.selected = PickInfo("scene", -1, scene_name, default_note)

        self.camera_yaw = math.radians(38.0)
        self.camera_pitch = math.radians(22.0)
        self.camera_distance = 12.5
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.drag_start: tuple[int, int] | None = None
        self.drag_mode: str | None = None

        self._configure_style()
        self._build_layout()
        self._draw_scene()
        self._refresh_inspector()
        self.after(16, self._step_simulation)

    def _infer_grid_shape(self, count: int) -> tuple[int, int]:
        if count == 0:
            return 0, 0
        side = int(round(math.sqrt(count)))
        if side * side == count:
            return side, side
        return count, 1

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure(
            "PanelHeader.TLabel",
            background=PANEL_BG,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "PanelBody.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "PanelValue.TLabel",
            background=PANEL_BG,
            foreground=TEXT_PRIMARY,
            font=("Consolas", 10),
        )
        style.configure(
            "Status.TLabel",
            background=BACKGROUND,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, style="Root.TFrame", padding=(16, 12))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        title = ttk.Label(toolbar, text=self.scene_name, style="PanelHeader.TLabel")
        title.grid(row=0, column=0, sticky="w")

        hint = ttk.Label(
            toolbar,
            text="Wheel zoom | Left drag orbit | Right drag pan | Click inspect | Space play/pause",
            style="Status.TLabel",
        )
        hint.grid(row=0, column=1, sticky="e")

        content = ttk.Frame(self, style="Root.TFrame", padding=(16, 0, 16, 16))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)

        viewport_shell = ttk.Frame(content, style="Panel.TFrame")
        viewport_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        viewport_shell.columnconfigure(0, weight=1)
        viewport_shell.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            viewport_shell,
            background=BACKGROUND,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._orbit_camera)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Button-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._pan_camera)
        self.canvas.bind("<ButtonRelease-3>", self._end_drag)
        self.canvas.bind("<MouseWheel>", self._zoom_view)
        self.bind("<space>", self._toggle_play)

        inspector = ttk.Frame(content, style="Panel.TFrame", padding=18, width=330)
        inspector.grid(row=0, column=1, sticky="ns")
        inspector.grid_propagate(False)
        inspector.columnconfigure(0, weight=1)

        ttk.Label(inspector, text="Inspector", style="PanelHeader.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            inspector,
            text="Viewer is a tool surface. Use --pipeline to choose a simulation, and add --viewer only when you want the visual inspector.",
            style="PanelBody.TLabel",
            wraplength=280,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 18))

        self.inspector_name = self._add_field(inspector, 2, "Name")
        self.inspector_type = self._add_field(inspector, 3, "Type")
        self.inspector_position = self._add_field(inspector, 4, "Position")
        self.inspector_state = self._add_field(inspector, 5, "State")
        self.inspector_note = self._add_field(inspector, 6, "Notes", wrap=True)

        ttk.Separator(inspector).grid(row=7, column=0, sticky="ew", pady=18)
        ttk.Label(inspector, text="Scene Stats", style="PanelHeader.TLabel").grid(
            row=8, column=0, sticky="w"
        )
        self.scene_stats = ttk.Label(
            inspector, style="PanelBody.TLabel", justify="left", wraplength=280
        )
        self.scene_stats.grid(row=9, column=0, sticky="ew", pady=(8, 0))

        status = ttk.Frame(self, style="Root.TFrame", padding=(16, 0, 16, 12))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)

        self.status_text = ttk.Label(status, style="Status.TLabel")
        self.status_text.grid(row=0, column=0, sticky="w")

    def _add_field(
        self, parent: ttk.Frame, row: int, title: str, wrap: bool = False
    ) -> ttk.Label:
        group = ttk.Frame(parent, style="Panel.TFrame")
        group.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        group.columnconfigure(0, weight=1)

        ttk.Label(group, text=title.upper(), style="PanelBody.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        value = ttk.Label(
            group,
            style="PanelValue.TLabel",
            wraplength=280 if wrap else 0,
            justify="left",
        )
        value.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        return value

    def _toggle_play(self, _event: tk.Event | None = None) -> None:
        if self.scene is None:
            return
        self.is_playing = not self.is_playing
        self._update_status()

    def _step_simulation(self) -> None:
        if self.scene is not None and self.is_playing and self.particles is not None:
            self.scene.step(self.dt)
            self.positions = self.particles.positions_numpy()
            self.velocities = self.particles.velocities_numpy()
            self.frame_index += 1
            self._draw_scene()
            self._refresh_inspector()
        self.after(16, self._step_simulation)

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        self._draw_background(width, height)
        self._draw_ground_plane()
        self._draw_grid()
        self._draw_axes()
        if len(self.positions) > 0:
            self._draw_particles()
        self._draw_overlay(width, height)
        self._update_status()

    def _draw_background(self, width: int, height: int) -> None:
        self.canvas.create_rectangle(0, 0, width, height, fill=SKY_BOTTOM, outline="")
        horizon = int(height * 0.44)
        self.canvas.create_rectangle(0, 0, width, horizon, fill=SKY_TOP, outline="")

    def _draw_ground_plane(self) -> None:
        corners: list[ScreenPoint | None] = [
            self.project_point(-8.0, 0.0, -8.0),
            self.project_point(8.0, 0.0, -8.0),
            self.project_point(8.0, 0.0, 8.0),
            self.project_point(-8.0, 0.0, 8.0),
        ]
        if any(point is None for point in corners):
            return

        projected_corners = [point for point in corners if point is not None]
        coords = [value for point in projected_corners for value in point]
        self.canvas.create_polygon(
            *coords, fill=GROUND_FILL, outline="#29303a", width=2
        )

    def _draw_grid(self) -> None:
        for value in range(-8, 9):
            color = GRID_MAJOR if value % 2 == 0 else GRID_MINOR
            width = 2 if value == 0 else 1
            self._draw_world_line(value, 0.0, -8.0, value, 0.0, 8.0, color, width)
            self._draw_world_line(-8.0, 0.0, value, 8.0, 0.0, value, color, width)

    def _draw_axes(self) -> None:
        self._draw_world_line(0.0, 0.0, 0.0, 1.8, 0.0, 0.0, AXIS_X, 3)
        self._draw_world_line(0.0, 0.0, 0.0, 0.0, 1.8, 0.0, AXIS_Y, 3)
        self._draw_world_line(0.0, 0.0, 0.0, 0.0, 0.0, 1.8, AXIS_Z, 3)

    def _draw_world_line(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        color: str,
        width: int,
    ) -> None:
        p1 = self.project_point(x1, y1, z1)
        p2 = self.project_point(x2, y2, z2)
        if p1 is None or p2 is None:
            return
        self.canvas.create_line(*p1, *p2, fill=color, width=width)

    def _draw_particles(self) -> None:
        projected: list[tuple[float, int, float, float]] = []
        for index, pos in enumerate(self.positions):
            point = self.project_point(float(pos[0]), float(pos[1]), float(pos[2]))
            if point is None:
                continue
            depth = self.camera_space(float(pos[0]), float(pos[1]), float(pos[2]))[2]
            projected.append((depth, index, point[0], point[1]))

        if self.grid_width > 1 and self.grid_height > 1:
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    current = row * self.grid_width + col
                    if col + 1 < self.grid_width:
                        self._draw_particle_edge(current, current + 1)
                    if row + 1 < self.grid_height:
                        self._draw_particle_edge(current, current + self.grid_width)

        projected.sort(reverse=True)
        for _, index, screen_x, screen_y in projected:
            self._draw_particle(index, screen_x, screen_y)

    def _draw_particle_edge(self, a: int, b: int) -> None:
        pa = self.positions[a]
        pb = self.positions[b]
        p0 = self.project_point(float(pa[0]), float(pa[1]), float(pa[2]))
        p1 = self.project_point(float(pb[0]), float(pb[1]), float(pb[2]))
        if p0 is None or p1 is None:
            return
        self.canvas.create_line(*p0, *p1, fill=CLOTH_LINE, width=1)

    def _draw_particle(self, index: int, screen_x: float, screen_y: float) -> None:
        pos = self.positions[index]
        depth = max(
            0.6, self.camera_space(float(pos[0]), float(pos[1]), float(pos[2]))[2]
        )
        radius = max(3.0, 10.0 / depth)
        is_pinned = len(self.inv_mass) > index and self.inv_mass[index] == 0.0
        is_selected = self.selected.kind == "particle" and self.selected.index == index
        fill = PINNED_POINT if is_pinned else CLOTH_POINT
        outline = (
            SELECTION if is_selected else (PINNED_POINT if is_pinned else CLOTH_LINE)
        )
        border_width = 3 if is_selected else 1
        self.canvas.create_oval(
            screen_x - radius,
            screen_y - radius,
            screen_x + radius,
            screen_y + radius,
            fill=fill,
            outline=outline,
            width=border_width,
        )

    def _draw_overlay(self, width: int, height: int) -> None:
        self.canvas.create_text(
            18,
            18,
            anchor="nw",
            text=f"Perspective / {self.scene_name}",
            fill=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        self.canvas.create_rectangle(
            width - 244, 16, width - 16, 84, fill="#171b20", outline="#39414a"
        )
        self.canvas.create_text(
            width - 130,
            32,
            text=f"Frame {self.frame_index:04d}",
            fill=TEXT_PRIMARY,
            font=("Consolas", 10),
        )
        state_text = (
            "Playing"
            if self.is_playing and self.scene is not None
            else ("Paused" if self.scene is not None else "Tool Idle")
        )
        self.canvas.create_text(
            width - 130, 50, text=state_text, fill=TEXT_PRIMARY, font=("Consolas", 10)
        )
        self.canvas.create_text(
            width - 130,
            68,
            text=f"CamDist {self.camera_distance:.2f}",
            fill=TEXT_MUTED,
            font=("Consolas", 9),
        )

    def _refresh_inspector(self) -> None:
        if self.selected.kind == "particle" and 0 <= self.selected.index < len(
            self.positions
        ):
            pos = self.positions[self.selected.index]
            velocity = self.velocities[self.selected.index]
            pinned = (
                len(self.inv_mass) > self.selected.index
                and self.inv_mass[self.selected.index] == 0.0
            )
            self.inspector_name.configure(text=self.selected.name)
            self.inspector_type.configure(text="Particle")
            self.inspector_position.configure(
                text=f"X {pos[0]:.3f}, Y {pos[1]:.3f}, Z {pos[2]:.3f}"
            )
            self.inspector_state.configure(
                text=f"vel {np.linalg.norm(velocity):.3f} | pinned {str(bool(pinned)).lower()}"
            )
            self.inspector_note.configure(text=self.selected.note)
        elif len(self.positions) > 0:
            center = self.positions.mean(axis=0)
            constraint_count = max(
                0,
                (self.grid_width - 1) * self.grid_height
                + (self.grid_height - 1) * self.grid_width,
            )
            self.inspector_name.configure(text=self.scene_name)
            self.inspector_type.configure(text="System")
            self.inspector_position.configure(
                text=f"Center X {center[0]:.3f}, Y {center[1]:.3f}, Z {center[2]:.3f}"
            )
            self.inspector_state.configure(
                text=f"particles {len(self.positions)} | constraints {constraint_count}"
            )
            self.inspector_note.configure(text=self.selected.note)
        else:
            self.inspector_name.configure(text=self.scene_name)
            self.inspector_type.configure(text="Viewer")
            self.inspector_position.configure(text="No pipeline attached")
            self.inspector_state.configure(text="Use --pipeline <name> --viewer")
            self.inspector_note.configure(text=self.selected.note)

        if len(self.positions) > 0:
            min_pos = self.positions.min(axis=0)
            max_pos = self.positions.max(axis=0)
            self.scene_stats.configure(
                text=(
                    f"Frame: {self.frame_index}\n"
                    f"Particles: {len(self.positions)}\n"
                    f"Pinned: {int(np.count_nonzero(self.inv_mass == 0.0))}\n"
                    f"Bounds Y: {min_pos[1]:.3f} .. {max_pos[1]:.3f}"
                )
            )
        else:
            self.scene_stats.configure(
                text=(
                    "Frame: 0\n"
                    "Particles: 0\n"
                    "Pipeline: none\n"
                    "Tip: python run.py --pipeline xpbd_cloth --viewer"
                )
            )
        self._update_status()

    def _update_status(self) -> None:
        selected_name = self.selected.name
        state = (
            "playing"
            if self.is_playing and self.scene is not None
            else ("paused" if self.scene is not None else "idle")
        )
        self.status_text.configure(
            text=(
                f"Selected: {selected_name}  |  "
                f"Frame: {self.frame_index}  |  "
                f"Viewer: {state}  |  "
                f"Camera target: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f})"
            )
        )

    def _camera_position(self) -> tuple[float, float, float]:
        cos_pitch = math.cos(self.camera_pitch)
        return (
            self.target_x
            + self.camera_distance * math.sin(self.camera_yaw) * cos_pitch,
            self.target_y + self.camera_distance * math.sin(self.camera_pitch),
            self.target_z
            + self.camera_distance * math.cos(self.camera_yaw) * cos_pitch,
        )

    def camera_space(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        cam_x, cam_y, cam_z = self._camera_position()
        dx = x - cam_x
        dy = y - cam_y
        dz = z - cam_z

        forward_x = self.target_x - cam_x
        forward_y = self.target_y - cam_y
        forward_z = self.target_z - cam_z
        forward_len = math.sqrt(
            forward_x * forward_x + forward_y * forward_y + forward_z * forward_z
        )
        forward_x /= forward_len
        forward_y /= forward_len
        forward_z /= forward_len

        right_x = forward_z
        right_y = 0.0
        right_z = -forward_x
        right_len = max(1e-6, math.sqrt(right_x * right_x + right_z * right_z))
        right_x /= right_len
        right_z /= right_len

        up_x = right_y * forward_z - right_z * forward_y
        up_y = right_z * forward_x - right_x * forward_z
        up_z = right_x * forward_y - right_y * forward_x

        return (
            dx * right_x + dy * right_y + dz * right_z,
            dx * up_x + dy * up_y + dz * up_z,
            dx * forward_x + dy * forward_y + dz * forward_z,
        )

    def project_point(self, x: float, y: float, z: float) -> ScreenPoint | None:
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        cam_x, cam_y, cam_z = self.camera_space(x, y, z)
        if cam_z <= 0.2:
            return None
        focal = min(width, height) * 0.9
        return width * 0.5 + (cam_x / cam_z) * focal, height * 0.56 + (
            cam_y / cam_z
        ) * focal

    def _pick_particle(self, screen_x: float, screen_y: float) -> int | None:
        if len(self.positions) == 0:
            return None
        best_index: int | None = None
        best_distance = float("inf")
        for index, pos in enumerate(self.positions):
            point = self.project_point(float(pos[0]), float(pos[1]), float(pos[2]))
            if point is None:
                continue
            distance = math.dist((screen_x, screen_y), point)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None or best_distance > 16.0:
            return None
        return best_index

    def _on_left_down(self, event: tk.Event) -> None:
        picked = self._pick_particle(event.x, event.y)
        if picked is not None:
            self.selected = PickInfo(
                "particle",
                picked,
                f"Particle {picked}",
                f"Particle view from {self.scene_name}.",
            )
            self._refresh_inspector()
            self._draw_scene()
            self.drag_mode = None
            self.drag_start = None
            return

        self.selected = PickInfo("scene", -1, self.scene_name, self.selected.note)
        self._refresh_inspector()
        self.drag_mode = "orbit"
        self.drag_start = (event.x, event.y)

    def _on_right_down(self, event: tk.Event) -> None:
        self.drag_mode = "pan"
        self.drag_start = (event.x, event.y)

    def _orbit_camera(self, event: tk.Event) -> None:
        if self.drag_mode != "orbit" or self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.camera_yaw += dx * 0.008
        self.camera_pitch = min(
            math.radians(80.0), max(math.radians(8.0), self.camera_pitch + dy * 0.006)
        )
        self.drag_start = (event.x, event.y)
        self._draw_scene()

    def _pan_camera(self, event: tk.Event) -> None:
        if self.drag_mode != "pan" or self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]

        pan_scale = 0.012 * self.camera_distance
        right_x = math.cos(self.camera_yaw)
        right_z = -math.sin(self.camera_yaw)
        forward_x = math.sin(self.camera_yaw)
        forward_z = math.cos(self.camera_yaw)

        self.target_x -= dx * pan_scale * right_x
        self.target_z -= dx * pan_scale * right_z
        self.target_x -= dy * pan_scale * forward_x * 0.35
        self.target_z -= dy * pan_scale * forward_z * 0.35
        self.target_y -= dy * pan_scale * 0.25

        self.drag_start = (event.x, event.y)
        self._draw_scene()

    def _end_drag(self, _event: tk.Event) -> None:
        self.drag_start = None
        self.drag_mode = None

    def _zoom_view(self, event: tk.Event) -> None:
        factor = 0.9 if event.delta > 0 else 1.1
        self.camera_distance = min(24.0, max(4.0, self.camera_distance * factor))
        self._draw_scene()


def launch_viewer(
    scene_factory: SceneFactory | None = None,
    title: str = "Dynames Viewer",
    scene_name: str = "Scene",
) -> None:
    app = DCCViewer(scene_factory=scene_factory, title=title, scene_name=scene_name)
    app.mainloop()


if __name__ == "__main__":
    launch_viewer()
