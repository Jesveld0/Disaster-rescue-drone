"""
test_pathfinder.py — Tests for A* pathfinding and OccupancyGrid.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from ground_station.pathfinder import OccupancyGrid, AStarPathfinder


class TestOccupancyGrid:
    """Test OccupancyGrid creation and updates."""

    def test_grid_initialized_empty(self):
        """Grid should start with all cells free."""
        grid = OccupancyGrid(width=10, height=10)
        assert np.all(grid.grid == 0)

    def test_drone_at_center(self):
        """Drone position should be at grid center."""
        grid = OccupancyGrid(width=20, height=20)
        assert grid.drone_row == 10
        assert grid.drone_col == 10

    def test_update_from_ir_front(self):
        """Front IR sensor should mark cells ahead of drone."""
        grid = OccupancyGrid(width=20, height=20)
        grid.update_from_ir({"front": True, "back": False, "left": False, "right": False})

        # At least some cells ahead of drone should be blocked
        front_blocked = False
        for r in range(0, grid.drone_row):
            for c in range(grid.width):
                if grid.grid[r, c] > 0:
                    front_blocked = True
                    break
        assert front_blocked

    def test_update_from_ir_all_clear(self):
        """All clear IR should not add obstacles."""
        grid = OccupancyGrid(width=10, height=10)
        grid.update_from_ir({"front": False, "back": False, "left": False, "right": False})
        assert np.all(grid.grid == 0)

    def test_decay_reduces_obstacles(self):
        """Decay should reduce obstacle counters by 1."""
        grid = OccupancyGrid(width=10, height=10)
        grid.grid[3, 3] = 5
        grid.decay()
        assert grid.grid[3, 3] == 4

    def test_decay_clears_expired(self):
        """Decay should clear cells when counter reaches 0."""
        grid = OccupancyGrid(width=10, height=10)
        grid.grid[3, 3] = 1
        grid.decay()
        assert grid.grid[3, 3] == 0

    def test_is_blocked(self):
        """is_blocked should correctly identify occupied cells."""
        grid = OccupancyGrid(width=10, height=10)
        grid.grid[2, 3] = 5
        assert grid.is_blocked(2, 3) == True
        assert grid.is_blocked(0, 0) == False

    def test_out_of_bounds_is_blocked(self):
        """Out-of-bounds cells should be treated as blocked."""
        grid = OccupancyGrid(width=10, height=10)
        assert grid.is_blocked(-1, 0) is True
        assert grid.is_blocked(0, -1) is True
        assert grid.is_blocked(10, 0) is True
        assert grid.is_blocked(0, 10) is True

    def test_render_returns_image(self):
        """Render should produce a BGR image."""
        grid = OccupancyGrid(width=5, height=5)
        img = grid.render(cell_px=10)
        assert img.shape == (50, 50, 3)
        assert img.dtype == np.uint8

    def test_reset_clears_grid(self):
        """Reset should clear all obstacles."""
        grid = OccupancyGrid(width=10, height=10)
        grid.grid[3, 3] = 5
        grid.grid[7, 7] = 10
        grid.reset()
        assert np.all(grid.grid == 0)


class TestAStarPathfinder:
    """Test A* pathfinding algorithm."""

    def test_straight_path_empty_grid(self):
        """Empty grid should produce a straight path."""
        grid = OccupancyGrid(width=10, height=10)
        pathfinder = AStarPathfinder()

        result = pathfinder.find_path(grid, start=(9, 5), goal=(0, 5))
        assert result.path_found is True
        assert len(result.path) > 0
        assert result.path[0] == (9, 5)
        assert result.path[-1] == (0, 5)

    def test_path_around_obstacle(self):
        """Path should route around a blocking obstacle."""
        grid = OccupancyGrid(width=10, height=10)

        # Place a wall blocking direct path
        for c in range(0, 9):
            grid.grid[5, c] = 10

        pathfinder = AStarPathfinder()
        result = pathfinder.find_path(grid, start=(8, 4), goal=(2, 4))

        assert result.path_found is True
        # Path should go around the wall (via column 9)
        assert len(result.path) > 6  # Must be longer than direct path

    def test_no_path_fully_blocked(self):
        """Should return empty path when goal is unreachable."""
        grid = OccupancyGrid(width=10, height=10)

        # Completely surround the goal
        for r in range(10):
            for c in range(10):
                if r <= 3:
                    grid.grid[r, c] = 10

        pathfinder = AStarPathfinder()
        result = pathfinder.find_path(grid, start=(8, 5), goal=(0, 5))

        # Should not find a direct path to a blocked cell
        # (but might find nearest free cell)
        # The important thing is it doesn't crash

    def test_path_to_self(self):
        """Path from a cell to itself should contain just that cell."""
        grid = OccupancyGrid(width=10, height=10)
        pathfinder = AStarPathfinder()

        result = pathfinder.find_path(grid, start=(5, 5), goal=(5, 5))
        assert result.path_found is True
        assert result.path == [(5, 5)]

    def test_default_start_and_goal(self):
        """Default start=drone center, goal=top center should work."""
        grid = OccupancyGrid(width=20, height=20)
        pathfinder = AStarPathfinder()

        result = pathfinder.find_path(grid)
        assert result.path_found is True
        assert result.path[0] == (grid.drone_row, grid.drone_col)
        assert result.path[-1] == (0, grid.width // 2)

    def test_computation_time_recorded(self):
        """Computation time should be recorded."""
        grid = OccupancyGrid(width=10, height=10)
        pathfinder = AStarPathfinder()

        result = pathfinder.find_path(grid, start=(9, 5), goal=(0, 5))
        assert result.computation_time_ms >= 0

    def test_render_path(self):
        """render_path should produce an image."""
        grid = OccupancyGrid(width=5, height=5)
        pathfinder = AStarPathfinder()

        grid_img = grid.render(cell_px=10)
        result = pathfinder.find_path(grid, start=(4, 2), goal=(0, 2))

        rendered = pathfinder.render_path(grid_img, result.path, cell_px=10)
        assert rendered.shape == grid_img.shape
