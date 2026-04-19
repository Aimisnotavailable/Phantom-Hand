# phantom_hand/__init__.py
"""
PhantomHand: MediaPipe hand tracking with ghost frames for occlusion handling.
"""

from .tracker import PhantomHandTracker

__version__ = "0.1.0"
__all__ = ["PhantomHandTracker"]