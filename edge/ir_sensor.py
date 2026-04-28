"""
ir_sensor.py — IR proximity sensor reader for the Raspberry Pi.

Reads 4 directional IR proximity sensors (front, back, left, right)
connected to GPIO pins. Includes debounce logic and a background
polling thread for real-time obstacle detection.

Digital IR sensors output LOW (0) when an obstacle is detected
and HIGH (1) when clear (active-low).
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import (
    IR_FRONT_PIN, IR_BACK_PIN, IR_LEFT_PIN, IR_RIGHT_PIN,
    IR_POLL_INTERVAL, IR_DEBOUNCE_COUNT,
)

logger = logging.getLogger(__name__)

# Try to import GPIO libraries (will fail on non-Pi hardware)
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logger.warning("RPi.GPIO not available — using simulated IR sensors")


@dataclass
class IRSensorState:
    """Current state of all IR proximity sensors."""
    front: bool = False   # True = obstacle detected
    back: bool = False
    left: bool = False
    right: bool = False

    def any_obstacle(self) -> bool:
        """Return True if any sensor detects an obstacle."""
        return self.front or self.back or self.left or self.right

    def as_dict(self) -> dict[str, bool]:
        """Return sensor state as a dictionary."""
        return {
            "front": self.front,
            "back": self.back,
            "left": self.left,
            "right": self.right,
        }


class IRSensorArray:
    """
    Manages 4 directional IR proximity sensors on the Raspberry Pi.

    Features:
    - Background thread polling at configurable intervals (default 50ms)
    - Debounce logic: requires N consecutive reads to confirm obstacle
    - Thread-safe access to latest sensor state
    - Graceful fallback to simulated mode when GPIO unavailable

    Usage:
        sensors = IRSensorArray()
        sensors.start()
        state = sensors.get_state()  # IRSensorState
        sensors.stop()
    """

    # Sensor configuration: name -> GPIO pin
    SENSORS = {
        "front": IR_FRONT_PIN,
        "back": IR_BACK_PIN,
        "left": IR_LEFT_PIN,
        "right": IR_RIGHT_PIN,
    }

    def __init__(self, simulated: bool = False):
        """
        Args:
            simulated: Force simulated mode (for testing without hardware).
        """
        self._simulated = simulated or not HAS_GPIO
        self._state = IRSensorState()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Debounce counters: how many consecutive detections per sensor
        self._debounce_counts = {name: 0 for name in self.SENSORS}

        if not self._simulated:
            self._setup_gpio()

    def _setup_gpio(self):
        """Initialize GPIO pins for sensor input."""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in self.SENSORS.items():
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                logger.info("IR sensor '%s' configured on GPIO %d", name, pin)
        except Exception as e:
            logger.error("GPIO setup failed: %s — falling back to simulated", e)
            self._simulated = True

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="IRSensorPoller"
        )
        self._thread.start()

        mode = "SIMULATED" if self._simulated else "HARDWARE"
        logger.info("IR sensor array started (%s mode)", mode)

    def _poll_loop(self):
        """Background thread: poll sensors at regular intervals."""
        while self._running:
            try:
                if self._simulated:
                    # In simulated mode, all sensors report clear
                    new_state = IRSensorState()
                else:
                    new_state = self._read_sensors()

                with self._lock:
                    self._state = new_state

            except Exception as e:
                logger.error("IR sensor poll error: %s", e)

            time.sleep(IR_POLL_INTERVAL)

    def _read_sensors(self) -> IRSensorState:
        """
        Read all GPIO pins with debounce logic.

        Digital IR sensors are typically active-low:
        - GPIO LOW (0) = obstacle detected
        - GPIO HIGH (1) = no obstacle
        """
        results = {}

        for name, pin in self.SENSORS.items():
            raw_value = GPIO.input(pin)
            obstacle_now = (raw_value == 0)  # Active-low

            if obstacle_now:
                self._debounce_counts[name] += 1
            else:
                self._debounce_counts[name] = 0

            # Require consecutive reads to confirm
            results[name] = self._debounce_counts[name] >= IR_DEBOUNCE_COUNT

        return IRSensorState(**results)

    def get_state(self) -> IRSensorState:
        """Get the current sensor state (thread-safe)."""
        with self._lock:
            return IRSensorState(
                front=self._state.front,
                back=self._state.back,
                left=self._state.left,
                right=self._state.right,
            )

    def set_simulated_state(
        self, front: bool = False, back: bool = False,
        left: bool = False, right: bool = False,
    ):
        """
        Manually set sensor state (for testing/simulation only).

        Only works when in simulated mode.
        """
        if not self._simulated:
            logger.warning("Cannot set simulated state on hardware sensors")
            return
        with self._lock:
            self._state = IRSensorState(
                front=front, back=back, left=left, right=right,
            )

    def stop(self):
        """Stop polling and clean up GPIO."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if not self._simulated and HAS_GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass

        logger.info("IR sensor array stopped")

    @property
    def is_simulated(self) -> bool:
        """Check if running in simulated mode."""
        return self._simulated
