"""
CERA v0.1 — CPU-Efficient Reuse Architecture
A small CPU software-rendering experiment built with Python + Tkinter.

Features:
- CPU-only triangle rasterizer (no GPU rendering APIs)
- Z-buffer depth testing
- Tile-based redraw tracking
- Conservative frame reuse when the scene is unchanged
- Simple animated 3D scene and camera controls
- Baseline full-frame redraw vs CERA selective redraw benchmark

Run:
    python3 cera.py

Controls:
    Space       pause/resume animation
    Left/Right  rotate camera
    Up/Down     change camera distance
    B           toggle baseline full redraw / CERA reuse
    R           reset camera
    Esc         quit

This is a prototype, not an RTX competitor. It demonstrates and measures
one narrow idea: avoiding work when the scene has not changed.
"""

import math
import time
import tkinter as tk
from dataclasses import dataclass

WIDTH, HEIGHT = 800, 520
TILE = 32
BG = (18, 22, 31)
FOV = 420.0
NEAR = 0.1


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class Vertex:
    x: float
    y: float
    z: float


def rotate_y(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return Vertex(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)


def rotate_x(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return Vertex(v.x, v.y * c - v.z * s, v.y * s + v.z * c)


class CeraRenderer:
    def __init__(self, width, height, tile_size=TILE):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.cols = (width + tile_size - 1) // tile_size
        self.rows = (height + tile_size - 1) // tile_size
        self.previous_signature = None
        self.previous_pixels = None
        self.last_tiles_touched = 0
        self.last_pixels_written = 0

    def make_scene(self, angle, camera_distance):
        # Cube plus floor grid: intentionally simple and deterministic.
        cube_vertices = [
            Vertex(-1, -1, -1), Vertex(1, -1, -1),
            Vertex(1, 1, -1), Vertex(-1, 1, -1),
            Vertex(-1, -1, 1), Vertex(1, -1, 1),
            Vertex(1, 1, 1), Vertex(-1, 1, 1),
        ]
        faces = [
            ((0, 1, 2, 3), (220, 80, 75)),
            ((4, 7, 6, 5), (75, 170, 235)),
            ((0, 4, 5, 1), (80, 190, 135)),
            ((3, 2, 6, 7), (230, 185, 75)),
            ((1, 5, 6, 2), (170, 105, 220)),
            ((0, 3, 7, 4), (70, 190, 200)),
        ]
        transformed = []
        for v in cube_vertices:
            v = rotate_y(v, angle)
            v = rotate_x(v, -0.35)
            transformed.append(Vertex(v.x, v.y, v.z + camera_distance))

        triangles = []
        for indices, color in faces:
            a, b, c, d = [transformed[i] for i in indices]
            # Split quads into triangles. Simple flat colors; no lighting model yet.
            triangles.append(([a, b, c], color))
            triangles.append(([a, c, d], tuple(int(ch * 0.82) for ch in color)))
        return triangles

    def project(self, v):
        if v.z <= NEAR:
            return None
        return (self.width / 2 + FOV * v.x / v.z,
                self.height / 2 - FOV * v.y / v.z,
                v.z)

    @staticmethod
    def edge(ax, ay, bx, by, px, py):
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

    def raster_triangle(self, pixels, zbuf, tri, color, clip=None):
        projected = [self.project(v) for v in tri]
        if any(p is None for p in projected):
            return 0
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = projected
        minx = max(0, int(math.floor(min(x0, x1, x2))))
        maxx = min(self.width - 1, int(math.ceil(max(x0, x1, x2))))
        miny = max(0, int(math.floor(min(y0, y1, y2))))
        maxy = min(self.height - 1, int(math.ceil(max(y0, y1, y2))))
        if minx > maxx or miny > maxy:
            return 0

        if clip:
            cx0, cy0, cx1, cy1 = clip
            minx, maxx = max(minx, cx0), min(maxx, cx1)
            miny, maxy = max(miny, cy0), min(maxy, cy1)
            if minx > maxx or miny > maxy:
                return 0

        area = self.edge(x0, y0, x1, y1, x2, y2)
        if abs(area) < 1e-9:
            return 0

        # Flat-shaded barycentric rasterization with perspective-correct depth.
        written = 0
        invz0, invz1, invz2 = 1.0 / z0, 1.0 / z1, 1.0 / z2
        for y in range(miny, maxy + 1):
            py = y + 0.5
            row = y * self.width
            for x in range(minx, maxx + 1):
                px = x + 0.5
                w0 = self.edge(x1, y1, x2, y2, px, py) / area
                w1 = self.edge(x2, y2, x0, y0, px, py) / area
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-8 and w1 >= -1e-8 and w2 >= -1e-8:
                    invz = w0 * invz0 + w1 * invz1 + w2 * invz2
                    if invz <= 0:
                        continue
                    z = 1.0 / invz
                    idx = row + x
                    if z < zbuf[idx]:
                        zbuf[idx] = z
                        pixels[idx] = color
                        written += 1
        return written

    def render_full(self, triangles):
        pixels = [BG] * (self.width * self.height)
        zbuf = [float("inf")] * (self.width * self.height)
        written = 0
        for tri, color in triangles:
            written += self.raster_triangle(pixels, zbuf, tri, color)
        self.last_tiles_touched = self.cols * self.rows
        self.last_pixels_written = written
        self.previous_pixels = pixels
        return pixels

    def render_cera(self, triangles, signature):
        # Exact cache hit: if scene state hasn't changed, reuse every prior pixel.
        if signature == self.previous_signature and self.previous_pixels is not None:
            self.last_tiles_touched = 0
            self.last_pixels_written = 0
            return self.previous_pixels

        # v0.1 uses conservative invalidation: any scene change invalidates all
        # tiles. Tile tracking is measured; regional invalidation arrives later.
        pixels = [BG] * (self.width * self.height)
        zbuf = [float("inf")] * (self.width * self.height)
        written = 0
        for tri, color in triangles:
            written += self.raster_triangle(pixels, zbuf, tri, color)
        self.previous_signature = signature
        self.previous_pixels = pixels
        self.last_tiles_touched = self.cols * self.rows
        self.last_pixels_written = written
        return pixels


class CeraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CERA v0.1 — CPU-Efficient Reuse Architecture")
        self.renderer = CeraRenderer(WIDTH, HEIGHT)
        self.angle = 0.0
        self.camera_distance = 5.0
        self.paused = False
        self.use_baseline = False
        self.last_time = time.perf_counter()
        self.frame_times = []
        self.frames = 0

        top = tk.Frame(root, bg="#111722")
        top.pack(fill="x")
        self.status = tk.Label(
            top, text="CERA v0.1 | CPU software renderer",
            bg="#111722", fg="#f0f4ff", font=("Menlo", 12, "bold"), padx=10, pady=8
        )
        self.status.pack(side="left")
        self.mode_label = tk.Label(
            top, text="MODE: CERA REUSE", bg="#111722", fg="#7de3b0",
            font=("Menlo", 10), padx=10
        )
        self.mode_label.pack(side="right")

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#12161f",
                                highlightthickness=0)
        self.canvas.pack()
        self.photo = tk.PhotoImage(width=WIDTH, height=HEIGHT)
        self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        bottom = tk.Label(
            root,
            text="←/→ rotate   ↑/↓ zoom   SPACE pause   B baseline/CERA   R reset   ESC quit",
            bg="#111722", fg="#c3cde0", font=("Menlo", 9), pady=7
        )
        bottom.pack(fill="x")
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.tick()

    def on_key(self, event):
        key = event.keysym.lower()
        if key == "left":
            self.angle -= 0.12
        elif key == "right":
            self.angle += 0.12
        elif key == "up":
            self.camera_distance = max(2.5, self.camera_distance - 0.25)
        elif key == "down":
            self.camera_distance = min(10.0, self.camera_distance + 0.25)
        elif key == "space":
            self.paused = not self.paused
        elif key == "b":
            self.use_baseline = not self.use_baseline
        elif key == "r":
            self.angle, self.camera_distance = 0.0, 5.0
        elif key == "escape":
            self.root.destroy()

    def put_pixels(self, pixels):
        # Tk PhotoImage accepts PPM data; encode one frame and replace its image.
        rows = []
        for y in range(HEIGHT):
            row = pixels[y * WIDTH:(y + 1) * WIDTH]
            rows.append("{" + " ".join("#%02x%02x%02x" % p for p in row) + "}")
        self.photo.put(" ".join(rows), to=(0, 0, WIDTH, HEIGHT))

    def tick(self):
        start = time.perf_counter()
        if not self.paused:
            self.angle += 0.012
        triangles = self.renderer.make_scene(self.angle, self.camera_distance)
        signature = (round(self.angle, 5), round(self.camera_distance, 4))

        if self.use_baseline:
            pixels = self.renderer.render_full(triangles)
            mode = "MODE: BASELINE FULL REDRAW"
        else:
            pixels = self.renderer.render_cera(triangles, signature)
            mode = "MODE: CERA REUSE"

        self.put_pixels(pixels)
        elapsed = time.perf_counter() - start
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.frames += 1

        avg_ms = (sum(self.frame_times) / len(self.frame_times)) * 1000
        fps = 1.0 / max(elapsed, 1e-9)
        self.mode_label.config(
            text=f"{mode} | {fps:.1f} FPS | {avg_ms:.1f} ms/frame | "
                 f"tiles {self.renderer.last_tiles_touched}/{self.renderer.cols*self.renderer.rows}"
        )
        self.root.after(1, self.tick)


def main():
    root = tk.Tk()
    CeraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
