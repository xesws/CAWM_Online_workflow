"""
File-based image locking system for preventing race conditions in Docker image operations.

When multiple processes run in parallel and try to pull the same Docker image,
race conditions can occur. This module provides a file-based locking mechanism
to ensure only one process pulls an image at a time.
"""

import hashlib
import os
import tempfile
import time
from typing import Optional

from openhands.core.logger import openhands_logger as logger

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class ImageLock:
    """File-based lock for a specific Docker image to prevent race conditions.

    This lock ensures that when multiple processes try to pull the same image,
    only one actually performs the pull while others wait.

    Usage:
        lock = ImageLock("my-image:latest")
        if lock.acquire():
            try:
                # pull image here
                pass
            finally:
                lock.release()

    Or using context manager:
        with ImageLock("my-image:latest"):
            # pull image here
            pass
    """

    def __init__(self, image_name: str, lock_dir: Optional[str] = None):
        """Initialize the image lock.

        Args:
            image_name: The Docker image name to lock (e.g., "ubuntu:latest")
            lock_dir: Optional directory for lock files. Defaults to system temp dir.
        """
        self.image_name = image_name
        self.lock_dir = lock_dir or os.path.join(
            tempfile.gettempdir(), 'openhands_image_locks'
        )
        # Use MD5 hash to create a safe filename from image name
        sanitized_name = hashlib.md5(image_name.encode()).hexdigest()
        self.lock_file_path = os.path.join(
            self.lock_dir, f'image_{sanitized_name}.lock'
        )
        self.lock_fd: Optional[int] = None
        self._locked = False

        # Ensure lock directory exists
        os.makedirs(self.lock_dir, exist_ok=True)

    def acquire(self, timeout: float = 300.0) -> bool:
        """Acquire the lock for this image.

        Args:
            timeout: Maximum time to wait for the lock in seconds.
                     Default is 5 minutes (image pulls can be slow).

        Returns:
            True if lock was acquired, False if timeout occurred.
        """
        if self._locked:
            return True

        try:
            self.lock_fd = os.open(
                self.lock_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC
            )

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if HAS_FCNTL:
                        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._locked = True
                    # Write debug info to lock file
                    os.write(
                        self.lock_fd, f'{self.image_name}\n{os.getpid()}\n'.encode()
                    )
                    os.fsync(self.lock_fd)
                    logger.debug(f'Acquired lock for image {self.image_name}')
                    return True
                except (OSError, IOError):
                    # Lock is held by another process, wait and retry
                    time.sleep(0.5)

            # Timeout occurred
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
            logger.warning(
                f'Timeout waiting for lock on image {self.image_name} after {timeout}s'
            )
            return False

        except Exception as e:
            logger.debug(f'Failed to acquire lock for image {self.image_name}: {e}')
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except OSError:
                    pass
                self.lock_fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self.lock_fd is not None:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                # Try to remove the lock file
                try:
                    os.unlink(self.lock_file_path)
                except FileNotFoundError:
                    pass
                logger.debug(f'Released lock for image {self.image_name}')
            except Exception as e:
                logger.warning(f'Error releasing lock for image {self.image_name}: {e}')
            finally:
                self.lock_fd = None
                self._locked = False

    def __enter__(self) -> 'ImageLock':
        """Context manager entry."""
        if not self.acquire():
            raise OSError(f'Could not acquire lock for image {self.image_name}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    @property
    def is_locked(self) -> bool:
        """Check if this lock instance currently holds the lock."""
        return self._locked
