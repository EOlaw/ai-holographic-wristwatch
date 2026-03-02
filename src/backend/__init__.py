"""
Backend Package — AI Holographic Wristwatch

Cloud and on-device backend services:

- api/               : REST, WebSocket, GraphQL endpoints
- services/          : Core business logic services (user, notification, sync)
- storage/           : Time-series health data, user data, model weights
- security/          : Authentication, encryption, privacy, zero-knowledge proofs
- monitoring/        : System health, performance metrics, error tracking
- data_processing/   : ETL pipelines, feature engineering, data normalization
- ai_infrastructure/ : Model serving, inference optimization, knowledge management

Architecture:
    Backend services run both on-device (local inference, edge storage)
    and in the cloud (heavy computation, long-term storage, multi-device sync).
    All services use dependency injection for testability.
    Data privacy: health data is end-to-end encrypted before cloud transmission.
"""

__version__ = "1.0.0"

__all__ = [
    "api",
    "services",
    "storage",
    "security",
    "monitoring",
    "data_processing",
    "ai_infrastructure",
]
