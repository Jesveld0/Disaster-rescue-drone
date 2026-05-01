"""
pathfinder.py — A* pathfinding integrated with the ground station pipeline.

Builds a 2D occupancy grid from live perception data (YOLO detections, thermal
fire zones, MiDaS depth) and computes a safe navigation path for the drone.

Grid convention (matches camera perspective):
    Row 0   = top of frame   (far ahead of drone)
    Row N-1 = bottom of frame (drone's immediate foreground)
    Start   = bottom-center  (drone's current position)
    Goal    = top-center (forward) OR toward the nearest person in fire
"""

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    PATHFINDING_GRID_COLS, PATHFINDING_GRID_ROWS,
    PATHFINDING_OBSTACLE_COST, PATHFINDING_PERSON_COST,
    PATHFINDING_FIRE_COST, PATHFINDING_DEPTH_COST_SCALE,
    PATHFINDING_DEPTH_THRESHOLD,
    RGB_WIDTH, RGB_HEIGHT,
)
from ground_station.detector import DetectionResult
from ground_station.fusion import FusionResult

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class PathfindingResult:
    """Result produced by the Pathfinder each inference cycle."""
    path: Optional[list[tuple[int, int]]]       # (row, col) grid waypoints
    path_pixels: Optional[list[tuple[int, int]]] # (x, y) pixel coords on frame
    start_grid: tuple[int, int]                  # grid (row, col)
    goal_grid: tuple[int, int]                   # grid (row, col)
    start_pixel: tuple[int, int]                 # pixel (x, y)
    goal_pixel: tuple[int, int]                  # pixel (x, y)
    compute_time_ms: float
    grid_snapshot: np.ndarray                    # uint8 cost grid for debug viz
    navigating_to_person: bool = False           # True when goal is a person in fire


# =============================================================================
# Occupancy Grid
# =============================================================================

class OccupancyGrid:
    """
    2D cost grid derived from live perception data.

    Cell costs:
        0        = free space
        1–99     = weighted (depth proximity / person zone)
        100–254  = high cost but passable
        255      = impassable (obstacle or fire)
    """

    def __init__(
        self,
        rows: int = PATHFINDING_GRID_ROWS,
        cols: int = PATHFINDING_GRID_COLS,
    ):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.uint8)

    def clear(self):
        self.grid[:] = 0

    def build(
        self,
        frame_shape: tuple,
        detections: DetectionResult,
        fusion_result: FusionResult,
        depth_map: Optional[np.ndarray] = None,
    ):
        """
        Populate the grid from ground station perception results.

        Priority (highest cost wins per cell):
            fire / obstacles → 255 (impassable)
            person zones     → 80  (high cost, passable — avoid flying over)
            depth proximity  → 1–50 added to existing cost
        """
        self.clear()
        frame_h, frame_w = frame_shape[:2]
        sx = self.cols / frame_w
        sy = self.rows / frame_h

        def _apply_bbox(bbox, cost):
            x1, y1, x2, y2 = bbox
            gx1 = max(0, int(x1 * sx))
            gy1 = max(0, int(y1 * sy))
            gx2 = min(self.cols, int(x2 * sx) + 1)
            gy2 = min(self.rows, int(y2 * sy) + 1)
            self.grid[gy1:gy2, gx1:gx2] = np.maximum(
                self.grid[gy1:gy2, gx1:gx2], cost
            )

        # Obstacles → impassable
        for obs in detections.obstacles:
            _apply_bbox(obs.bbox, PATHFINDING_OBSTACLE_COST)

        # Fire zones (thermally confirmed or visual) → impassable
        for fz in fusion_result.fire_zones:
            _apply_bbox(fz.bbox, PATHFINDING_FIRE_COST)

        # Persons → high cost but passable (prefer not to fly over people)
        for person in detections.persons:
            _apply_bbox(person.bbox, PATHFINDING_PERSON_COST)

        # Depth-based cost (additive, only on currently passable cells)
        if depth_map is not None:
            depth_resized = cv2.resize(
                depth_map, (self.cols, self.rows),
                interpolation=cv2.INTER_AREA,
            )
            above = np.clip(depth_resized - PATHFINDING_DEPTH_THRESHOLD, 0, 1)
            range_ = max(1.0 - PATHFINDING_DEPTH_THRESHOLD, 1e-6)
            depth_cost = (above / range_ * PATHFINDING_DEPTH_COST_SCALE).astype(np.uint8)

            passable = self.grid < PATHFINDING_OBSTACLE_COST
            combined = np.clip(
                self.grid.astype(np.int16) + depth_cost.astype(np.int16), 0, 254
            ).astype(np.uint8)
            self.grid[passable] = combined[passable]

    def is_passable(self, row: int, col: int) -> bool:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(self.grid[row, col]) < PATHFINDING_OBSTACLE_COST
        return False

    def get_cost(self, row: int, col: int) -> float:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return float("inf")
        val = int(self.grid[row, col])
        if val >= PATHFINDING_OBSTACLE_COST:
            return float("inf")
        return 1.0 + val / 10.0


# =============================================================================
# A* Pathfinder
# =============================================================================

@dataclass(order=True)
class _Node:
    f_score: float
    pos: tuple = field(compare=False)
    g_score: float = field(compare=False)


class AStarPathfinder:
    """
    A* search on an OccupancyGrid with 8-directional movement.

    Uses the octile distance heuristic (admissible + consistent for 8-dir).
    Weighted cells are respected — the planner prefers low-cost routes.
    """

    DIRECTIONS = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, 1.414),
        (-1,  1, 1.414),
        ( 1, -1, 1.414),
        ( 1,  1, 1.414),
    ]

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_iterations: int = 50000,
    ) -> Optional[list[tuple[int, int]]]:
        if not self.grid.is_passable(*start):
            logger.debug("A* start %s is impassable", start)
            return None
        if not self.grid.is_passable(*goal):
            logger.debug("A* goal %s is impassable", goal)
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

        for _ in range(max_iterations):
            if not open_set:
                break

            current = heapq.heappop(open_set)
            if current.pos == goal:
                return self._reconstruct(came_from, goal)
            if current.pos in closed:
                continue
            closed.add(current.pos)

            cr, cc = current.pos
            for dr, dc, move_cost in self.DIRECTIONS:
                neighbor = (cr + dr, cc + dc)
                if neighbor in closed:
                    continue
                cell_cost = self.grid.get_cost(*neighbor)
                if cell_cost == float("inf"):
                    continue
                tentative_g = current.g_score + move_cost * cell_cost
                if tentative_g < g_scores.get(neighbor, float("inf")):
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current.pos
                    heapq.heappush(open_set, _Node(
                        f_score=tentative_g + self._heuristic(neighbor, goal),
                        pos=neighbor,
                        g_score=tentative_g,
                    ))

        return None

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple, tuple], goal: tuple
    ) -> list[tuple[int, int]]:
        path, current = [goal], goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# Pathfinder Coordinator
# =============================================================================

class Pathfinder:
    """
    High-level pathfinding coordinator for the ground station pipeline.

    Wraps OccupancyGrid + AStarPathfinder and handles:
    - Grid construction from typed ground station data
    - Automatic start/goal selection each frame
    - Grid-to-pixel coordinate conversion
    - Path smoothing via waypoint decimation
    """

    def __init__(
        self,
        grid_rows: int = PATHFINDING_GRID_ROWS,
        grid_cols: int = PATHFINDING_GRID_COLS,
        frame_width: int = RGB_WIDTH,
        frame_height: int = RGB_HEIGHT,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._grid = OccupancyGrid(rows=grid_rows, cols=grid_cols)
        self._astar = AStarPathfinder(self._grid)

    def update(
        self,
        detections: DetectionResult,
        fusion_result: FusionResult,
        depth_map: Optional[np.ndarray],
        frame_shape: Optional[tuple] = None,
    ) -> PathfindingResult:
        """
        Rebuild the occupancy grid and compute a fresh path.

        Args:
            detections: YOLO detection results.
            fusion_result: Thermal-RGB fusion results.
            depth_map: Optional normalized MiDaS depth map.
            frame_shape: (H, W[, C]) of source frame; falls back to config dims.

        Returns:
            PathfindingResult with path in both grid and pixel coordinates.
        """
        if frame_shape is None:
            frame_shape = (self.frame_height, self.frame_width)

        t0 = time.monotonic()

        # Build occupancy grid
        self._grid.build(frame_shape, detections, fusion_result, depth_map)

        # Determine start (bottom-center = drone's position)
        start = (self.grid_rows - 1, self.grid_cols // 2)
        start = self._find_nearest_passable(start)

        # Determine goal
        goal, navigating_to_person = self._select_goal(fusion_result, detections)
        goal = self._find_nearest_passable(goal)

        # Run A*
        path = None
        if start is not None and goal is not None:
            self._astar.grid = self._grid
            path = self._astar.find_path(start, goal)
            if path:
                path = self._decimate_path(path)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Convert to pixel coords
        path_pixels = None
        if path:
            path_pixels = [self._grid_to_pixel(r, c) for r, c in path]

        if start is None:
            start = (self.grid_rows - 1, self.grid_cols // 2)
        if goal is None:
            goal = (0, self.grid_cols // 2)

        result = PathfindingResult(
            path=path,
            path_pixels=path_pixels,
            start_grid=start,
            goal_grid=goal,
            start_pixel=self._grid_to_pixel(*start),
            goal_pixel=self._grid_to_pixel(*goal),
            compute_time_ms=round(elapsed_ms, 1),
            grid_snapshot=self._grid.grid.copy(),
            navigating_to_person=navigating_to_person,
        )

        logger.debug(
            "Pathfinding: %s in %.1f ms | path=%s steps",
            "found" if path else "no path",
            elapsed_ms,
            len(path) if path else 0,
        )

        return result

    def _select_goal(
        self,
        fusion_result: FusionResult,
        detections: DetectionResult,
    ) -> tuple[tuple[int, int], bool]:
        """
        Choose the navigation goal based on current detections.

        Returns:
            (goal_grid_pos, navigating_to_person)
        """
        # Navigate toward persons in fire (rescue priority)
        persons_in_fire = [p for p in fusion_result.persons if p.in_fire]
        if persons_in_fire:
            # Use the person with the highest temperature (most critical)
            target = max(persons_in_fire, key=lambda p: p.max_temp)
            x1, y1, x2, y2 = target.detection.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            goal = self._pixel_to_grid(cx, cy)
            return goal, True

        # Default: navigate forward (top-center of frame)
        return (0, self.grid_cols // 2), False

    def _find_nearest_passable(
        self, pos: tuple[int, int], max_radius: int = 8
    ) -> Optional[tuple[int, int]]:
        """
        Find the nearest passable cell to pos using BFS spiral search.

        Returns None only if the entire grid is impassable.
        """
        r0, c0 = pos
        r0 = max(0, min(self.grid_rows - 1, r0))
        c0 = max(0, min(self.grid_cols - 1, c0))

        if self._grid.is_passable(r0, c0):
            return (r0, c0)

        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    nr, nc = r0 + dr, c0 + dc
                    if self._grid.is_passable(nr, nc):
                        return (nr, nc)
        return None

    def _grid_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        x = int((col + 0.5) * self.frame_width / self.grid_cols)
        y = int((row + 0.5) * self.frame_height / self.grid_rows)
        return (x, y)

    def _pixel_to_grid(self, px: int, py: int) -> tuple[int, int]:
        col = int(px * self.grid_cols / self.frame_width)
        row = int(py * self.grid_rows / self.frame_height)
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return (row, col)

    @staticmethod
    def _decimate_path(
        path: list[tuple[int, int]], step: int = 3
    ) -> list[tuple[int, int]]:
        """Thin the path to every Nth waypoint for cleaner rendering."""
        if len(path) <= 2:
            return path
        decimated = path[::step]
        if decimated[-1] != path[-1]:
            decimated.append(path[-1])
        return decimated
