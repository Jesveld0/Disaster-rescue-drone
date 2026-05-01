"""
pathfinding_demo.py — Standalone A* pathfinding demo for disaster rescue drone.

Builds a 2D occupancy grid from perception data (YOLO detections, MiDaS depth,
thermal fire zones) and computes safe navigation paths using A* search.

Three modes of operation:
    --simulate    Procedurally generated obstacle field (no models needed)
    --image FILE  Static image analyzed with YOLO + optional MiDaS
    (default)     Live webcam feed with real-time path recomputation

Keyboard controls:
    r  — Random start/goal positions
    s  — Toggle simulate overlay
    g  — Re-generate simulated obstacles
    c  — Clear path
    1  — Set click mode to START
    2  — Set click mode to GOAL
    q  — Quit

Mouse:
    Left-click to place start or goal (toggle with 1/2 keys)

Usage:
    python pathfinding_demo.py --simulate
    python pathfinding_demo.py --image test_scene.jpg
    python pathfinding_demo.py
    python pathfinding_demo.py --grid-size 80 60
"""

import argparse
import heapq
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, ".")

from config import (
    PATHFINDING_GRID_COLS, PATHFINDING_GRID_ROWS,
    PATHFINDING_OBSTACLE_COST, PATHFINDING_PERSON_COST,
    PATHFINDING_FIRE_COST, PATHFINDING_DEPTH_COST_SCALE,
    PATHFINDING_DEPTH_THRESHOLD,
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    MIDAS_MODEL_TYPE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Occupancy Grid
# =============================================================================

class OccupancyGrid:
    """
    2D cost grid derived from perception data.

    Cell values:
        0       = free space
        1-99    = weighted cost (prefer lower)
        255     = impassable
    """

    def __init__(self, rows: int = PATHFINDING_GRID_ROWS,
                 cols: int = PATHFINDING_GRID_COLS):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.uint8)
        # Separate layers for visualization
        self._obstacle_mask = np.zeros((rows, cols), dtype=bool)
        self._person_mask = np.zeros((rows, cols), dtype=bool)
        self._fire_mask = np.zeros((rows, cols), dtype=bool)
        self._depth_cost = np.zeros((rows, cols), dtype=np.float32)

    def clear(self):
        """Reset the grid to all-free."""
        self.grid[:] = 0
        self._obstacle_mask[:] = False
        self._person_mask[:] = False
        self._fire_mask[:] = False
        self._depth_cost[:] = 0

    def from_detections(
        self,
        frame_shape: tuple,
        detections: list,
        depth_map: Optional[np.ndarray] = None,
    ):
        """
        Build the occupancy grid from YOLO detections and depth data.

        Args:
            frame_shape: (height, width) of the source frame.
            detections: List of dicts with keys:
                'bbox': (x1, y1, x2, y2), 'category': str, 'class_name': str
            depth_map: Optional normalized depth map (H, W), 0-1.
        """
        self.clear()
        frame_h, frame_w = frame_shape[:2]
        scale_x = self.cols / frame_w
        scale_y = self.rows / frame_h

        # Mark detections on the grid
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Convert to grid coordinates
            gx1 = max(0, int(x1 * scale_x))
            gy1 = max(0, int(y1 * scale_y))
            gx2 = min(self.cols, int(x2 * scale_x) + 1)
            gy2 = min(self.rows, int(y2 * scale_y) + 1)

            category = det["category"]

            if category == "fire":
                self._fire_mask[gy1:gy2, gx1:gx2] = True
                self.grid[gy1:gy2, gx1:gx2] = np.maximum(
                    self.grid[gy1:gy2, gx1:gx2], PATHFINDING_FIRE_COST
                )
            elif category == "person":
                self._person_mask[gy1:gy2, gx1:gx2] = True
                self.grid[gy1:gy2, gx1:gx2] = np.maximum(
                    self.grid[gy1:gy2, gx1:gx2], PATHFINDING_PERSON_COST
                )
            elif category == "obstacle":
                self._obstacle_mask[gy1:gy2, gx1:gx2] = True
                self.grid[gy1:gy2, gx1:gx2] = np.maximum(
                    self.grid[gy1:gy2, gx1:gx2], PATHFINDING_OBSTACLE_COST
                )

        # Apply depth-based costs
        if depth_map is not None:
            depth_resized = cv2.resize(
                depth_map, (self.cols, self.rows),
                interpolation=cv2.INTER_AREA,
            )
            # Cells with high depth (close objects) get elevated cost
            depth_above = np.clip(
                depth_resized - PATHFINDING_DEPTH_THRESHOLD, 0, 1
            )
            depth_cost = (
                depth_above / (1.0 - PATHFINDING_DEPTH_THRESHOLD + 1e-6)
                * PATHFINDING_DEPTH_COST_SCALE
            ).astype(np.uint8)

            self._depth_cost = depth_above
            # Only add cost where not already impassable
            free_cells = self.grid < PATHFINDING_OBSTACLE_COST
            combined = np.clip(
                self.grid.astype(np.int16) + depth_cost.astype(np.int16),
                0, 254
            ).astype(np.uint8)
            self.grid[free_cells] = combined[free_cells]

    def from_simulated(self, num_obstacles: int = 15, num_fires: int = 5,
                       num_persons: int = 4, seed: Optional[int] = None):
        """
        Generate a procedural obstacle field for testing.

        Creates random rectangular obstacles, fire zones, and person regions.
        """
        self.clear()
        rng = np.random.RandomState(seed)

        def _place_rects(count, min_size, max_size, cost, mask):
            for _ in range(count):
                w = rng.randint(min_size, max_size + 1)
                h = rng.randint(min_size, max_size + 1)
                x = rng.randint(0, max(1, self.cols - w))
                y = rng.randint(0, max(1, self.rows - h))
                mask[y:y+h, x:x+w] = True
                self.grid[y:y+h, x:x+w] = np.maximum(
                    self.grid[y:y+h, x:x+w], cost
                )

        # Add obstacles (large blockers)
        _place_rects(num_obstacles, 2, 6, PATHFINDING_OBSTACLE_COST,
                     self._obstacle_mask)

        # Add fire zones (medium)
        _place_rects(num_fires, 2, 5, PATHFINDING_FIRE_COST,
                     self._fire_mask)

        # Add person regions (small, high-cost but passable)
        _place_rects(num_persons, 1, 3, PATHFINDING_PERSON_COST,
                     self._person_mask)

        # Add some depth gradient cost (simulated proximity)
        gradient = np.zeros((self.rows, self.cols), dtype=np.float32)
        for _ in range(3):
            cx = rng.randint(0, self.cols)
            cy = rng.randint(0, self.rows)
            yy, xx = np.mgrid[:self.rows, :self.cols]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            radius = rng.randint(5, 15)
            blob = np.clip(1.0 - dist / radius, 0, 1) * 0.4
            gradient = np.maximum(gradient, blob)

        self._depth_cost = gradient
        depth_addition = (gradient * PATHFINDING_DEPTH_COST_SCALE).astype(np.uint8)
        free_cells = self.grid < PATHFINDING_OBSTACLE_COST
        combined = np.clip(
            self.grid.astype(np.int16) + depth_addition.astype(np.int16),
            0, 254
        ).astype(np.uint8)
        self.grid[free_cells] = combined[free_cells]

    def is_passable(self, row: int, col: int) -> bool:
        """Check if a cell is passable (not impassable)."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] < PATHFINDING_OBSTACLE_COST
        return False

    def get_cost(self, row: int, col: int) -> float:
        """Get traversal cost for a cell. Returns inf for impassable."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return float("inf")
        val = self.grid[row, col]
        if val >= PATHFINDING_OBSTACLE_COST:
            return float("inf")
        return 1.0 + val / 10.0  # Base cost 1.0 + penalty


# =============================================================================
# A* Pathfinder
# =============================================================================

@dataclass(order=True)
class _Node:
    """Priority queue node for A*."""
    f_score: float
    pos: tuple = field(compare=False)
    g_score: float = field(compare=False)


class AStarPathfinder:
    """
    A* pathfinding on a 2D occupancy grid.

    Supports 8-directional movement with diagonal cost √2.
    Weighted cells are respected — the pathfinder prefers low-cost routes.
    """

    # 8 directions: (dr, dc, cost_multiplier)
    DIRECTIONS = [
        (-1,  0, 1.0),   # up
        ( 1,  0, 1.0),   # down
        ( 0, -1, 1.0),   # left
        ( 0,  1, 1.0),   # right
        (-1, -1, 1.414), # up-left
        (-1,  1, 1.414), # up-right
        ( 1, -1, 1.414), # down-left
        ( 1,  1, 1.414), # down-right
    ]

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_iterations: int = 50000,
    ) -> Optional[list[tuple[int, int]]]:
        """
        Find the shortest weighted path from start to goal.

        Args:
            start: (row, col) start position.
            goal: (row, col) goal position.
            max_iterations: Safety limit to prevent infinite loops.

        Returns:
            List of (row, col) waypoints from start to goal, or None if
            no path exists.
        """
        if not self.grid.is_passable(*start):
            logger.warning("Start position %s is impassable", start)
            return None
        if not self.grid.is_passable(*goal):
            logger.warning("Goal position %s is impassable", goal)
            return None

        open_set: list[_Node] = []
        heapq.heappush(open_set, _Node(
            f_score=self._heuristic(start, goal),
            pos=start,
            g_score=0.0,
        ))

        came_from: dict[tuple, tuple] = {}
        g_scores: dict[tuple, float] = {start: 0.0}
        closed: set[tuple] = set()
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)

            if current.pos == goal:
                return self._reconstruct_path(came_from, goal)

            if current.pos in closed:
                continue
            closed.add(current.pos)

            cr, cc = current.pos
            for dr, dc, move_cost in self.DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                if neighbor in closed:
                    continue

                cell_cost = self.grid.get_cost(nr, nc)
                if cell_cost == float("inf"):
                    continue

                tentative_g = current.g_score + move_cost * cell_cost

                if tentative_g < g_scores.get(neighbor, float("inf")):
                    g_scores[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current.pos
                    heapq.heappush(open_set, _Node(
                        f_score=f, pos=neighbor, g_score=tentative_g,
                    ))

        logger.info(
            "A* completed: %d iterations, path %s",
            iterations, "found" if goal in came_from else "NOT found",
        )
        return None  # No path found

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Octile distance heuristic (consistent for 8-dir movement)."""
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple, tuple], goal: tuple
    ) -> list[tuple[int, int]]:
        """Trace back from goal to start through came_from map."""
        path = [goal]
        current = goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# Path Visualizer
# =============================================================================

class PathVisualizer:
    """
    Renders the occupancy grid, path, and camera frame side-by-side.

    Color coding:
        Dark gray     = free space
        Orange        = obstacles (impassable)
        Red           = fire zones (impassable)
        Cyan          = person zones (high cost, passable)
        Yellow-Orange = depth-based elevated cost
        Bright green  = computed A* path
        Green circle  = start position
        Red circle    = goal position
    """

    WINDOW_NAME = "Pathfinding Demo — Disaster Rescue Drone"

    # Colors (BGR)
    COLOR_FREE = (40, 40, 40)
    COLOR_OBSTACLE = (0, 140, 255)    # Orange
    COLOR_FIRE = (0, 0, 220)          # Red
    COLOR_PERSON = (220, 200, 0)      # Cyan
    COLOR_PATH = (0, 255, 100)        # Bright green
    COLOR_START = (0, 255, 0)         # Green
    COLOR_GOAL = (0, 0, 255)          # Red
    COLOR_DEPTH = (0, 200, 255)       # Yellow-ish
    COLOR_EXPLORED = (60, 60, 60)     # Slightly lighter gray
    COLOR_GRID_LINE = (55, 55, 55)    # Subtle grid

    def __init__(self, grid: OccupancyGrid, cell_size: int = 12):
        self.grid = grid
        self.cell_size = cell_size
        self.grid_img_w = grid.cols * cell_size
        self.grid_img_h = grid.rows * cell_size

        # Click mode: 'start' or 'goal'
        self.click_mode = "start"

        # Positions
        self.start_pos: Optional[tuple[int, int]] = None
        self.goal_pos: Optional[tuple[int, int]] = None
        self.path: Optional[list[tuple[int, int]]] = None
        self.path_time_ms: float = 0.0

        # Stats
        self.total_cells = grid.rows * grid.cols
        self.free_cells = 0
        self.blocked_cells = 0

    def render_grid(self) -> np.ndarray:
        """Render the occupancy grid as a color-coded image."""
        cs = self.cell_size
        img = np.full((self.grid_img_h, self.grid_img_w, 3),
                       self.COLOR_FREE, dtype=np.uint8)

        # Draw depth cost gradient first (background layer)
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                y1, y2 = r * cs, (r + 1) * cs
                x1, x2 = c * cs, (c + 1) * cs

                depth_val = self.grid._depth_cost[r, c]
                if depth_val > 0.01 and not self.grid._obstacle_mask[r, c] \
                        and not self.grid._fire_mask[r, c]:
                    # Blend toward yellow based on depth cost
                    alpha = min(depth_val * 2, 1.0)
                    base = np.array(self.COLOR_FREE, dtype=np.float32)
                    target = np.array(self.COLOR_DEPTH, dtype=np.float32)
                    blended = (base * (1 - alpha) + target * alpha).astype(np.uint8)
                    img[y1:y2, x1:x2] = blended

        # Draw categorical overlays
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                y1, y2 = r * cs, (r + 1) * cs
                x1, x2 = c * cs, (c + 1) * cs

                if self.grid._fire_mask[r, c]:
                    img[y1:y2, x1:x2] = self.COLOR_FIRE
                elif self.grid._obstacle_mask[r, c]:
                    img[y1:y2, x1:x2] = self.COLOR_OBSTACLE
                elif self.grid._person_mask[r, c]:
                    img[y1:y2, x1:x2] = self.COLOR_PERSON

        # Draw subtle grid lines
        for r in range(self.grid.rows + 1):
            y = r * cs
            cv2.line(img, (0, y), (self.grid_img_w, y),
                     self.COLOR_GRID_LINE, 1)
        for c in range(self.grid.cols + 1):
            x = c * cs
            cv2.line(img, (x, 0), (x, self.grid_img_h),
                     self.COLOR_GRID_LINE, 1)

        # Draw path
        if self.path and len(self.path) >= 2:
            for i in range(len(self.path) - 1):
                r1, c1 = self.path[i]
                r2, c2 = self.path[i + 1]
                pt1 = (c1 * cs + cs // 2, r1 * cs + cs // 2)
                pt2 = (c2 * cs + cs // 2, r2 * cs + cs // 2)
                cv2.line(img, pt1, pt2, self.COLOR_PATH, max(2, cs // 4))

            # Draw path dots
            for r, c in self.path:
                center = (c * cs + cs // 2, r * cs + cs // 2)
                cv2.circle(img, center, max(2, cs // 6),
                           self.COLOR_PATH, -1)

        # Draw start marker
        if self.start_pos is not None:
            sr, sc = self.start_pos
            center = (sc * cs + cs // 2, sr * cs + cs // 2)
            cv2.circle(img, center, cs // 2 + 2, self.COLOR_START, -1)
            cv2.circle(img, center, cs // 2 + 2, (255, 255, 255), 2)
            cv2.putText(img, "S", (center[0] - 4, center[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)

        # Draw goal marker
        if self.goal_pos is not None:
            gr, gc = self.goal_pos
            center = (gc * cs + cs // 2, gr * cs + cs // 2)
            cv2.circle(img, center, cs // 2 + 2, self.COLOR_GOAL, -1)
            cv2.circle(img, center, cs // 2 + 2, (255, 255, 255), 2)
            cv2.putText(img, "G", (center[0] - 4, center[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return img

    def render_info_panel(self, width: int, height: int) -> np.ndarray:
        """Render an info panel with stats and controls."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        line_h = 28

        def put(text, color=(200, 200, 200), scale=0.55, thick=1):
            nonlocal y
            cv2.putText(panel, text, (15, y), font, scale, color, thick,
                        cv2.LINE_AA)
            y += line_h

        # Title
        cv2.putText(panel, "A* PATHFINDING", (15, y), font, 0.75,
                     (0, 255, 100), 2, cv2.LINE_AA)
        y += 40

        # Stats
        self.free_cells = int(np.sum(self.grid.grid < PATHFINDING_OBSTACLE_COST))
        self.blocked_cells = self.total_cells - self.free_cells

        put(f"Grid: {self.grid.cols} x {self.grid.rows}")
        put(f"Free cells: {self.free_cells}", (100, 255, 100))
        put(f"Blocked cells: {self.blocked_cells}", (100, 100, 255))

        y += 10
        if self.start_pos:
            put(f"Start: ({self.start_pos[1]}, {self.start_pos[0]})",
                self.COLOR_START)
        else:
            put("Start: not set", (100, 100, 100))

        if self.goal_pos:
            put(f"Goal: ({self.goal_pos[1]}, {self.goal_pos[0]})",
                self.COLOR_GOAL)
        else:
            put("Goal: not set", (100, 100, 100))

        y += 10
        if self.path is not None:
            put(f"Path length: {len(self.path)} steps", (0, 255, 100))
            put(f"Compute time: {self.path_time_ms:.1f} ms", (200, 200, 200))
        else:
            if self.start_pos and self.goal_pos:
                put("No path found!", (0, 0, 255))
            else:
                put("Set start & goal", (100, 100, 100))

        # Controls legend
        y += 20
        put("--- Controls ---", (150, 150, 150), 0.5)
        put(f"Click mode: [{self.click_mode.upper()}]",
            self.COLOR_START if self.click_mode == "start" else self.COLOR_GOAL)
        put("1 = set START mode", (150, 150, 150), 0.45)
        put("2 = set GOAL mode", (150, 150, 150), 0.45)
        put("r = random positions", (150, 150, 150), 0.45)
        put("g = regenerate grid", (150, 150, 150), 0.45)
        put("c = clear path", (150, 150, 150), 0.45)
        put("q = quit", (150, 150, 150), 0.45)

        # Legend
        y += 20
        put("--- Legend ---", (150, 150, 150), 0.5)
        cv2.rectangle(panel, (15, y - 10), (30, y + 2),
                      self.COLOR_OBSTACLE, -1)
        cv2.putText(panel, "Obstacle", (40, y), font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h

        cv2.rectangle(panel, (15, y - 10), (30, y + 2),
                      self.COLOR_FIRE, -1)
        cv2.putText(panel, "Fire zone", (40, y), font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h

        cv2.rectangle(panel, (15, y - 10), (30, y + 2),
                      self.COLOR_PERSON, -1)
        cv2.putText(panel, "Person (avoid)", (40, y), font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h

        cv2.rectangle(panel, (15, y - 10), (30, y + 2),
                      self.COLOR_DEPTH, -1)
        cv2.putText(panel, "Depth cost", (40, y), font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h

        cv2.rectangle(panel, (15, y - 10), (30, y + 2),
                      self.COLOR_PATH, -1)
        cv2.putText(panel, "Safe path", (40, y), font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return panel

    def compose_display(self, camera_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compose the final display image.

        Layout: [camera frame (optional)] | [occupancy grid] | [info panel]
        """
        grid_img = self.render_grid()
        panel_width = 250
        info_panel = self.render_info_panel(panel_width, self.grid_img_h)

        if camera_frame is not None:
            # Resize camera frame to match grid height
            cam_h = self.grid_img_h
            cam_w = int(camera_frame.shape[1] * cam_h / camera_frame.shape[0])
            cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))

            # Draw detection boxes on camera frame
            display = np.hstack([cam_resized, grid_img, info_panel])
        else:
            display = np.hstack([grid_img, info_panel])

        return display

    def pixel_to_grid(self, px: int, py: int,
                      camera_width: int = 0) -> Optional[tuple[int, int]]:
        """
        Convert pixel coordinates (from mouse click) to grid coordinates.

        Args:
            px, py: Pixel coordinates in the composite display.
            camera_width: Width of the camera panel (0 if no camera).

        Returns:
            (row, col) grid coordinates or None if outside grid area.
        """
        # Offset by camera panel width
        gx = px - camera_width
        gy = py

        if gx < 0 or gx >= self.grid_img_w or gy < 0 or gy >= self.grid_img_h:
            return None

        col = gx // self.cell_size
        row = gy // self.cell_size

        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            return (row, col)
        return None

    def compute_path(self):
        """Run A* and store the result."""
        if self.start_pos is None or self.goal_pos is None:
            self.path = None
            return

        pathfinder = AStarPathfinder(self.grid)
        t0 = time.monotonic()
        self.path = pathfinder.find_path(self.start_pos, self.goal_pos)
        self.path_time_ms = (time.monotonic() - t0) * 1000

        if self.path:
            logger.info(
                "Path found: %d steps in %.1f ms",
                len(self.path), self.path_time_ms
            )
        else:
            logger.info("No path found (%.1f ms)", self.path_time_ms)

    def random_positions(self):
        """Set random passable start and goal positions."""
        passable = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.is_passable(r, c):
                    passable.append((r, c))

        if len(passable) < 2:
            logger.warning("Not enough passable cells for start/goal")
            return

        indices = np.random.choice(len(passable), size=2, replace=False)
        self.start_pos = passable[indices[0]]
        self.goal_pos = passable[indices[1]]
        self.compute_path()


# =============================================================================
# YOLO + MiDaS Integration (for webcam / image modes)
# =============================================================================

def load_yolo_detector():
    """Load YOLOv8 detector. Returns model or None."""
    try:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(YOLO_MODEL_PATH)
        # Warm up
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        model.predict(dummy, device=device, imgsz=320, verbose=False)
        logger.info("YOLO loaded on %s", device)
        return model, device
    except Exception as e:
        logger.error("Could not load YOLO: %s", e)
        return None, "cpu"


def load_midas():
    """Load MiDaS depth estimator. Returns (model, transform, device) or Nones."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE,
                               trust_repo=True)
        model.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                    trust_repo=True)
        transform = transforms.small_transform
        logger.info("MiDaS loaded on %s", device)
        return model, transform, device
    except Exception as e:
        logger.error("Could not load MiDaS: %s", e)
        return None, None, None


def run_yolo(model, device, frame: np.ndarray) -> list[dict]:
    """Run YOLO detection and return list of detection dicts."""
    if model is None:
        return []

    OBSTACLE_NAMES = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "bench", "chair", "potted plant", "couch", "bed",
        "dining table", "tv", "refrigerator",
    }

    detections = []
    try:
        results = model.predict(
            frame, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD,
            device=device, imgsz=320, verbose=False,
        )
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, bbox)
                name = names.get(cls_id, f"cls_{cls_id}")

                if cls_id == 0:
                    cat = "person"
                elif name.lower() in ("fire", "flame", "smoke"):
                    cat = "fire"
                elif name.lower() in OBSTACLE_NAMES:
                    cat = "obstacle"
                else:
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "category": cat,
                    "class_name": name,
                    "confidence": float(conf),
                })
    except Exception as e:
        logger.error("YOLO inference error: %s", e)

    return detections


def run_midas(model, transform, device, frame: np.ndarray) -> Optional[np.ndarray]:
    """Run MiDaS depth estimation. Returns normalized depth map or None."""
    if model is None:
        return None
    try:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 0:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)
    except Exception as e:
        logger.error("MiDaS inference error: %s", e)
        return None


def draw_detections_on_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detection bounding boxes on a camera frame."""
    display = frame.copy()
    CAT_COLORS = {
        "obstacle": (0, 140, 255),
        "fire": (0, 0, 220),
        "person": (220, 200, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CAT_COLORS.get(det["category"], (200, 200, 200))
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(display, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return display


# =============================================================================
# Main Application Loop
# =============================================================================

def run_simulate_mode(grid_rows: int, grid_cols: int):
    """Run with procedurally generated obstacles — no AI models needed."""
    print("\n" + "=" * 60)
    print("  A* Pathfinding Demo — SIMULATE MODE")
    print("  No AI models needed. Random obstacle field.")
    print("=" * 60 + "\n")

    grid = OccupancyGrid(rows=grid_rows, cols=grid_cols)
    grid.from_simulated()

    viz = PathVisualizer(grid)
    viz.random_positions()

    camera_w = 0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos = viz.pixel_to_grid(x, y, camera_width=camera_w)
            if pos is not None:
                if viz.click_mode == "start":
                    viz.start_pos = pos
                    logger.info("Start set to %s", pos)
                else:
                    viz.goal_pos = pos
                    logger.info("Goal set to %s", pos)
                viz.compute_path()

    cv2.namedWindow(PathVisualizer.WINDOW_NAME)
    cv2.setMouseCallback(PathVisualizer.WINDOW_NAME, on_mouse)

    while True:
        display = viz.compose_display(camera_frame=None)
        cv2.imshow(PathVisualizer.WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            viz.random_positions()
        elif key == ord("g"):
            grid.from_simulated(seed=int(time.time()) % 10000)
            viz.compute_path()
        elif key == ord("c"):
            viz.path = None
            viz.start_pos = None
            viz.goal_pos = None
        elif key == ord("1"):
            viz.click_mode = "start"
            logger.info("Click mode: START")
        elif key == ord("2"):
            viz.click_mode = "goal"
            logger.info("Click mode: GOAL")

    cv2.destroyAllWindows()


def run_camera_mode(grid_rows: int, grid_cols: int, image_path: Optional[str] = None):
    """Run with live webcam or static image + YOLO + optional MiDaS."""
    mode_name = "IMAGE" if image_path else "WEBCAM"
    print("\n" + "=" * 60)
    print(f"  A* Pathfinding Demo — {mode_name} MODE")
    print("  Loading YOLO + MiDaS models...")
    print("=" * 60 + "\n")

    yolo_model, yolo_device = load_yolo_detector()
    midas_model, midas_transform, midas_device = load_midas()

    grid = OccupancyGrid(rows=grid_rows, cols=grid_cols)
    viz = PathVisualizer(grid)

    # Open source
    if image_path:
        static_frame = cv2.imread(image_path)
        if static_frame is None:
            print(f"ERROR: Could not read image: {image_path}")
            return
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Try --simulate mode.")
            return
        static_frame = None

    camera_w = 0  # Will be computed after first frame

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos = viz.pixel_to_grid(x, y, camera_width=camera_w)
            if pos is not None:
                if viz.click_mode == "start":
                    viz.start_pos = pos
                    logger.info("Start set to %s", pos)
                else:
                    viz.goal_pos = pos
                    logger.info("Goal set to %s", pos)
                viz.compute_path()

    cv2.namedWindow(PathVisualizer.WINDOW_NAME)
    cv2.setMouseCallback(PathVisualizer.WINDOW_NAME, on_mouse)

    frame_count = 0
    last_detections = []

    while True:
        # Get frame
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = static_frame.copy()

        # Run detection every few frames (performance)
        if frame_count % 3 == 0 or frame_count == 0:
            last_detections = run_yolo(yolo_model, yolo_device, frame)
            depth_map = run_midas(midas_model, midas_transform,
                                  midas_device, frame)
            grid.from_detections(frame.shape, last_detections, depth_map)
            viz.compute_path()

        # Draw detections on camera frame
        cam_display = draw_detections_on_frame(frame, last_detections)

        # Compute camera panel width for mouse offset
        cam_h = viz.grid_img_h
        camera_w = int(cam_display.shape[1] * cam_h / cam_display.shape[0])

        display = viz.compose_display(camera_frame=cam_display)
        cv2.imshow(PathVisualizer.WINDOW_NAME, display)

        key = cv2.waitKey(1 if cap else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            viz.random_positions()
        elif key == ord("c"):
            viz.path = None
            viz.start_pos = None
            viz.goal_pos = None
        elif key == ord("1"):
            viz.click_mode = "start"
        elif key == ord("2"):
            viz.click_mode = "goal"

        frame_count += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A* Pathfinding Demo for Disaster Rescue Drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pathfinding_demo.py --simulate          # No models needed
  python pathfinding_demo.py                     # Live webcam + YOLO
  python pathfinding_demo.py --image scene.jpg   # Static image + YOLO
  python pathfinding_demo.py --grid-size 80 60   # Custom grid resolution
        """,
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use procedurally generated obstacles (no AI models needed)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a static image to analyze",
    )
    parser.add_argument(
        "--grid-size", nargs=2, type=int,
        default=[PATHFINDING_GRID_COLS, PATHFINDING_GRID_ROWS],
        metavar=("COLS", "ROWS"),
        help=f"Grid resolution (default: {PATHFINDING_GRID_COLS} {PATHFINDING_GRID_ROWS})",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    grid_cols, grid_rows = args.grid_size

    if args.simulate:
        run_simulate_mode(grid_rows, grid_cols)
    else:
        run_camera_mode(grid_rows, grid_cols, image_path=args.image)


if __name__ == "__main__":
    main()
