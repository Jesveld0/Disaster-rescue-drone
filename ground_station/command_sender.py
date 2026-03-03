"""
command_sender.py — UDP command sender from ground station to drone.

Sends compact command packets (SAFE/SLOW/STOP/FIRE_ALERT/HUMAN_IN_FIRE)
back to the drone via UDP.
"""

import logging
import socket
from typing import Optional

from config import COMMAND_PORT, COMMAND_NAMES
from protocol import encode_command

logger = logging.getLogger(__name__)


class CommandSender:
    """
    Sends command packets from the ground station to the drone.

    Maintains a UDP socket and tracks the last sent command to avoid
    flooding with duplicate commands.
    """

    def __init__(self, command_port: int = COMMAND_PORT):
        self.command_port = command_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_command: Optional[int] = None
        self._drone_address: Optional[tuple] = None
        self._send_count = 0

    def set_drone_address(self, address: tuple):
        """
        Set the drone's network address for command delivery.

        Args:
            address: (ip, port) tuple — typically obtained from the receiver.
        """
        # Use the source IP from the drone, but our command port
        self._drone_address = (address[0], self.command_port)
        logger.info("Drone command target set to %s:%d", address[0], self.command_port)

    def send(self, frame_id: int, command_code: int, force: bool = False):
        """
        Send a command packet to the drone.

        Args:
            frame_id: Current frame ID for correlation.
            command_code: Command code (0=SAFE, 1=SLOW, 2=STOP, 3=FIRE_ALERT, 4=HUMAN_IN_FIRE).
            force: Send even if same as last command.
        """
        if self._drone_address is None:
            logger.warning("Cannot send command — drone address not set")
            return

        # Deduplicate unless forced
        if not force and command_code == self._last_command:
            return

        cmd_name = COMMAND_NAMES.get(command_code, f"UNKNOWN({command_code})")

        try:
            packet = encode_command(frame_id, command_code)
            self.socket.sendto(packet, self._drone_address)
            self._last_command = command_code
            self._send_count += 1

            # Log critical commands at warning level
            if command_code >= 2:
                logger.warning(
                    "⚡ Sent command: %s (code=%d) to %s [frame %d]",
                    cmd_name, command_code, self._drone_address, frame_id,
                )
            else:
                logger.debug(
                    "Sent command: %s (code=%d) [frame %d]",
                    cmd_name, command_code, frame_id,
                )

        except OSError as e:
            logger.error("Failed to send command %s: %s", cmd_name, e)

    def send_stop(self, frame_id: int = 0):
        """Convenience: send STOP command immediately."""
        self.send(frame_id, 2, force=True)

    def send_safe(self, frame_id: int = 0):
        """Convenience: send SAFE command."""
        self.send(frame_id, 0)

    @property
    def last_command(self) -> Optional[int]:
        """Return the last command code sent."""
        return self._last_command

    def close(self):
        """Close the command socket."""
        self.socket.close()
        logger.info("CommandSender closed. Total commands sent: %d", self._send_count)
