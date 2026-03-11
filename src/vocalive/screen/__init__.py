from .base import ScreenCaptureEngine
from .macos import MacOSWindowScreenCapture

__all__ = [
    "MacOSWindowScreenCapture",
    "ScreenCaptureEngine",
]
