"""
servo_drop.py — Servo-controlled payload drop mechanism for Raspberry Pi.

Controls a sub-servo motor connected to GPIO 18 (hardware PWM) to
release a payload on command. The servo sweeps from closed (locked)
to open (dropped) position, holds, then returns.

Wiring:
    Servo signal wire → GPIO 18 (BCM)
    Servo VCC        → 5V pin
    Servo GND        → GND pin

Standard servo duty cycles at 50 Hz:
    2.5%  ≈  0° (full left)
    7.5%  ≈ 90° (neutral / closed / locked)
    12.5% ≈ 180° (full right / open / drop)

Adjust SERVO_CLOSED_DUTY and SERVO_OPEN_DUTY in config.py to match
your specific servo and drop mechanism geometry.
"""

import logging
import threading
import time

from config import (
    SERVO_GPIO_PIN, SERVO_FREQ_HZ,
    SERVO_CLOSED_DUTY, SERVO_OPEN_DUTY,
    SERVO_HOLD_SEC, SERVO_RETURN_DELAY,
)

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logger.warning("RPi.GPIO not available — servo running in simulated mode")


class ServoDrop:
    """
    Servo-controlled payload drop mechanism.

    Thread-safe: drop() can be called from any thread and executes
    the open → hold → close sequence in a background thread so it
    never blocks the camera capture loop.

    States:
        CLOSED  — locked, payload held, ready to drop
        DROPPING — servo moving to open position
        OPEN    — payload released, holding
        RETURNING — servo returning to closed
        COOLDOWN — brief pause after returning before accepting next drop

    Usage:
        servo = ServoDrop()
        servo.start()
        servo.drop()   # non-blocking
        servo.stop()
    """

    STATE_CLOSED = "CLOSED"
    STATE_DROPPING = "DROPPING"
    STATE_OPEN = "OPEN"
    STATE_RETURNING = "RETURNING"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(self, simulated: bool = False):
        self._simulated = simulated or not HAS_GPIO
        self._pwm = None
        self._state = self.STATE_CLOSED
        self._state_lock = threading.Lock()
        self._drop_thread: threading.Thread | None = None
        self._drop_count = 0

    def start(self):
        """Initialize GPIO and move servo to closed position."""
        if not self._simulated:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(SERVO_GPIO_PIN, GPIO.OUT)
                self._pwm = GPIO.PWM(SERVO_GPIO_PIN, SERVO_FREQ_HZ)
                self._pwm.start(SERVO_CLOSED_DUTY)
                time.sleep(0.3)  # let servo settle to closed
                self._pwm.ChangeDutyCycle(0)  # stop PWM jitter while idle
                logger.info(
                    "ServoDrop started: GPIO=%d  freq=%dHz  closed=%.1f%%  open=%.1f%%",
                    SERVO_GPIO_PIN, SERVO_FREQ_HZ, SERVO_CLOSED_DUTY, SERVO_OPEN_DUTY,
                )
            except Exception as e:
                logger.error("Servo GPIO init failed: %s — switching to simulated", e)
                self._simulated = True
        else:
            logger.info("ServoDrop started (SIMULATED mode)")

    def drop(self) -> bool:
        """
        Trigger payload drop (non-blocking).

        Returns True if drop was initiated, False if busy.
        The sequence runs in a background thread:
            1. Move to OPEN position
            2. Hold SERVO_HOLD_SEC seconds
            3. Return to CLOSED position
            4. Brief cooldown
        """
        with self._state_lock:
            if self._state != self.STATE_CLOSED:
                logger.warning(
                    "Drop request ignored — servo is busy (%s)", self._state
                )
                return False
            self._state = self.STATE_DROPPING

        self._drop_thread = threading.Thread(
            target=self._drop_sequence, daemon=True, name="ServoDropSequence"
        )
        self._drop_thread.start()
        return True

    def _drop_sequence(self):
        """Full open → hold → close cycle (runs in background thread)."""
        try:
            logger.info("🪂 DROP initiated (sequence #%d)", self._drop_count + 1)

            # 1. Open / release
            self._set_state(self.STATE_DROPPING)
            self._set_angle(SERVO_OPEN_DUTY)
            time.sleep(0.3)  # time for servo to reach open position

            # 2. Hold open
            self._set_state(self.STATE_OPEN)
            logger.info("Servo OPEN — payload released. Holding %.1fs", SERVO_HOLD_SEC)
            time.sleep(SERVO_HOLD_SEC)

            # 3. Return to closed
            self._set_state(self.STATE_RETURNING)
            self._set_angle(SERVO_CLOSED_DUTY)
            time.sleep(0.3)  # time for servo to reach closed position

            # 4. Stop PWM jitter, cooldown
            if not self._simulated and self._pwm:
                self._pwm.ChangeDutyCycle(0)

            self._set_state(self.STATE_COOLDOWN)
            self._drop_count += 1
            logger.info("Servo CLOSED — drop #%d complete", self._drop_count)
            time.sleep(SERVO_RETURN_DELAY)

            self._set_state(self.STATE_CLOSED)

        except Exception as e:
            logger.error("Drop sequence error: %s", e)
            self._set_state(self.STATE_CLOSED)

    def _set_angle(self, duty: float):
        """Set servo duty cycle (or log in simulated mode)."""
        if self._simulated:
            label = "OPEN" if duty == SERVO_OPEN_DUTY else "CLOSED"
            logger.info("[SIM] Servo → %s (duty=%.1f%%)", label, duty)
        else:
            if self._pwm:
                self._pwm.ChangeDutyCycle(duty)

    def _set_state(self, state: str):
        with self._state_lock:
            self._state = state

    @property
    def state(self) -> str:
        """Current servo state string."""
        with self._state_lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        """True if servo is in CLOSED state and ready to drop."""
        return self.state == self.STATE_CLOSED

    @property
    def drop_count(self) -> int:
        """Total number of successful drops."""
        return self._drop_count

    def stop(self):
        """Stop PWM and clean up GPIO."""
        if self._drop_thread and self._drop_thread.is_alive():
            self._drop_thread.join(timeout=SERVO_HOLD_SEC + 2.0)

        if not self._simulated and self._pwm:
            self._pwm.stop()

        if not self._simulated and HAS_GPIO:
            try:
                GPIO.cleanup(SERVO_GPIO_PIN)
            except Exception:
                pass

        logger.info("ServoDrop stopped. Total drops: %d", self._drop_count)
