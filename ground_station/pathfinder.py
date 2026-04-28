"""
pathfinder.py — A* pathfinding with occupancy grid for the ground station.

Maintains a 2D occupancy grid updated from IR sensor data and visual
detections. Computes optimal paths using A* algorithm with 8-directional
movement. All processing runs on the ground station.
"""

import logging
import heapq
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    GRID_WIDTH, GRID_HEIGHT, GRID_CELL_SIZE,
    OBSTACLE_DECAY_FRAMES, RGB_WIDTH, RGB_HEIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """Result of a pathfinding computation."""
    path: list[tuple[int, int]] = field(default_factory=list)  # [(row, col), ...]
    path_found: bool = False
    path_length: int = 0
    computation_time_ms: float = 0.0


class OccupancyGrid:
    """
    2D occupancy grid representing the environment around the drone.

    The drone is positioned at the center of the grid. Each cell can be:
    - 0: Free space
    - >0: Obstacle (value = remaining decay frames)

    Updated from two sources:
    1. IR proximity sensors — mark cells directly adjacent to the drone
    2. Visual detections — project obstacle bounding box centers into grid cells

    Obstacles decay over time (OBSTACLE_DECAY_FRAMES) to handle moving objects.
    """

    # Direction → grid offset for IR sensor mapping
    # Drone is at center; sensors mark 3 cells in the respective direction
    IR_DIRECTION_OFFSETS = {
        "front": [(-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 0), (-2, 1), (-3, 0)],
        "back":  [(1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (3, 0)],
        "left":  [(-1, -1), (0, -1), (1, -1), (-1, -2), (0, -2), (1, -2), (0, -3)],
        "right": [(-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2), (0, 3)],
    }

    def __init__(
        self,
        width: int = GRID_WIDTH,
        height: int = GRID_HEIGHT,
        cell_size: float = GRID_CELL_SIZE,
    ):
        """
        Args:
            width: Grid columns.
            height: Grid rows.
            cell_size: Physical size of each cell in meters.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Grid values: 0 = free, >0 = obstacle (decay counter)
        self.grid = np.zeros((height, width), dtype=np.int32)

        # Drone position is always at the center
        self.drone_row = height // 2
        self.drone_col = width // 2

    def update_from_ir(self, ir_state: dict[str, bool]):
        """
        Update grid cells based on IR proximity sensor readings.

        Args:
            ir_state: Dict with keys 'front', 'back', 'left', 'right'
                      and boolean values indicating obstacle presence.
        """
        for direction, has_obstacle in ir_state.items():
            if direction not in self.IR_DIRECTION_OFFSETS:
                continue

            offsets = self.IR_DIRECTION_OFFSETS[direction]
            for dr, dc in offsets:
                r = self.drone_row + dr
                c = self.drone_col + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    if has_obstacle:
                        self.grid[r, c] = OBSTACLE_DECAY_FRAMES
                    # Don't clear if not detected — let decay handle it

    def update_from_detections(
        self,
        obstacles: list,
        frame_width: int = RGB_WIDTH,
        frame_height: int = RGB_HEIGHT,
    ):
        """
        Project detected obstacle positions into the grid.

        Maps bounding box center coordinates from image space to grid space.
        Objects detected on the image edges are placed further from the drone.

        Args:
            obstacles: List of Detection objects with bbox attribute.
            frame_width: RGB frame width in pixels.
            frame_height: RGB frame height in pixels.
        """
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle.bbox

            # Bounding box center in normalized coordinates (0-1)
            cx = ((x1 + x2) / 2) / frame_width
            cy = ((y1 + y2) / 2) / frame_height

            # Map to grid coordinates
            # Horizontal: 0 (left edge) → 0, 0.5 (center) → drone_col, 1 (right edge) → width-1
            # Vertical: 0 (top/far) → 0, 1 (bottom/close) → height-1
            grid_col = int(cx * (self.width - 1))
            grid_row = int(cy * (self.height - 1))

            grid_col = max(0, min(self.width - 1, grid_col))
            grid_row = max(0, min(self.height - 1, grid_row))

            # Mark the cell and adjacent cells as obstacles
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r = grid_row + dr
                    c = grid_col + dc
                    if 0 <= r < self.height and 0 <= c < self.width:
                        self.grid[r, c] = OBSTACLE_DECAY_FRAMES

    def decay(self):
        """
        Reduce obstacle decay counters by 1.

        Called once per frame. When a counter reaches 0, the cell becomes free.
        This handles moving obstacles naturally.
        """
        mask = self.grid > 0
        self.grid[mask] -= 1

    def is_blocked(self, row: int, col: int) -> bool:
        """Check if a grid cell is occupied."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] > 0
        return True  # Out of bounds = blocked

    def reset(self):
        """Clear all obstacles from the grid."""
        self.grid.fill(0)

    def render(self, cell_px: int = 20) -> np.ndarray:
        """
        Render the grid as a BGR image for visualization.

        Args:
            cell_px: Pixel size of each grid cell.

        Returns:
            BGR image (height * cell_px, width * cell_px, 3).
        """
        h = self.height * cell_px
        w = self.width * cell_px
        img = np.zeros((h, w, 3), dtype=np.uint8)

        for r in range(self.height):
            for c in range(self.width):
                x1 = c * cell_px
                y1 = r * cell_px
                x2 = x1 + cell_px
                y2 = y1 + cell_px

                if self.grid[r, c] > 0:
                    # Obstacle — red, intensity based on decay
                    intensity = int(255 * self.grid[r, c] / OBSTACLE_DECAY_FRAMES)
                    color = (0, 0, intensity)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                else:
                    # Free — dark gray
                    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)

                # Grid lines
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

        # Draw drone position (green circle)
        dx = self.drone_col * cell_px + cell_px // 2
        dy = self.drone_row * cell_px + cell_px // 2
        cv2.circle(img, (dx, dy), cell_px // 3, (0, 255, 0), -1)

        return img


class AStarPathfinder:
    """
    A* pathfinding algorithm on the occupancy grid.

    Supports 8-directional movement with diagonal cost √2.
    Drone starts at the grid center and pathfinds to a specified goal.
    """

    # 8-directional movement: (dr, dc, cost)
    DIRECTIONS = [
        (-1, 0, 1.0),   # up
        (1, 0, 1.0),    # down
        (0, -1, 1.0),   # left
        (0, 1, 1.0),    # right
        (-1, -1, 1.414),  # up-left
        (-1, 1, 1.414),   # up-right
        (1, -1, 1.414),   # down-left
        (1, 1, 1.414),    # down-right
    ]

    def find_path(
        self,
        grid: OccupancyGrid,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
    ) -> PathResult:
        """
        Find the shortest path from start to goal using A*.

        Args:
            grid: OccupancyGrid with obstacle information.
            start: (row, col) start position. Defaults to drone position.
            goal: (row, col) goal position. Defaults to top-center (forward).

        Returns:
            PathResult with the computed path.
        """
        import time
        t0 = time.monotonic()

        if start is None:
            start = (grid.drone_row, grid.drone_col)

        if goal is None:
            # Default goal: move forward (top of grid)
            goal = (0, grid.width // 2)

        # Validate start and goal
        if grid.is_blocked(start[0], start[1]):
            return PathResult(computation_time_ms=(time.monotonic() - t0) * 1000)

        if grid.is_blocked(goal[0], goal[1]):
            # Try to find nearest free cell to goal
            goal = self._find_nearest_free(grid, goal)
            if goal is None:
                return PathResult(computation_time_ms=(time.monotonic() - t0) * 1000)

        # A* search
        path = self._astar(grid, start, goal)

        result = PathResult(
            path=path,
            path_found=len(path) > 0,
            path_length=len(path),
            computation_time_ms=round((time.monotonic() - t0) * 1000, 2),
        )

        return result

    def _astar(
        self,
        grid: OccupancyGrid,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Core A* algorithm implementation."""
        open_set = []  # (f_score, counter, node)
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        f_score: dict[tuple[int, int], float] = {start: self._heuristic(start, goal)}

        closed_set: set[tuple[int, int]] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for dr, dc, move_cost in self.DIRECTIONS:
                neighbor = (current[0] + dr, current[1] + dc)

                # Bounds check
                if not (0 <= neighbor[0] < grid.height and 0 <= neighbor[1] < grid.width):
                    continue

                # Obstacle check
                if grid.is_blocked(neighbor[0], neighbor[1]):
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        # No path found
        return []

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _find_nearest_free(
        grid: OccupancyGrid,
        target: tuple[int, int],
    ) -> Optional[tuple[int, int]]:
        """Find the nearest free cell to the target using BFS."""
        from collections import deque

        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([target])

        while queue:
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)

            r, c = cell
            if 0 <= r < grid.height and 0 <= c < grid.width:
                if not grid.is_blocked(r, c):
                    return cell

                # Explore neighbors
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited:
                            queue.append((nr, nc))

        return None

    def render_path(
        self,
        grid_img: np.ndarray,
        path: list[tuple[int, int]],
        cell_px: int = 20,
    ) -> np.ndarray:
        """
        Draw the computed path onto a grid visualization.

        Args:
            grid_img: Grid image from OccupancyGrid.render().
            path: List of (row, col) positions.
            cell_px: Pixel size of each grid cell.

        Returns:
            Grid image with path drawn in cyan.
        """
        img = grid_img.copy()

        if len(path) < 2:
            return img

        # Draw path lines
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            pt1 = (c1 * cell_px + cell_px // 2, r1 * cell_px + cell_px // 2)
            pt2 = (c2 * cell_px + cell_px // 2, r2 * cell_px + cell_px // 2)
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Cyan line

        # Draw goal marker
        goal_r, goal_c = path[-1]
        gx = goal_c * cell_px + cell_px // 2
        gy = goal_r * cell_px + cell_px // 2
        cv2.circle(img, (gx, gy), cell_px // 3, (0, 255, 255), -1)  # Yellow

        return img
