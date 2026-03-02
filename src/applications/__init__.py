"""
Applications Package — AI Holographic Wristwatch

End-user applications and developer tooling:

- watch_firmware/       : Core firmware loop, OS integration, hardware abstraction
- development_tools/    : Debugging, profiling, deployment, and QA tooling

Architecture:
    Applications are the topmost layer — they consume hardware, AI, and backend.
    The watch firmware runs the main event loop dispatching sensor events to AI.
    Development tools provide introspection and testing without modifying production.
"""

__version__ = "1.0.0"

__all__ = [
    "watch_firmware",
    "development_tools",
]
