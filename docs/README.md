# AI Holographic Wristwatch — Developer Implementation Guide

> **Senior-Level Project Reference:** This document maps every folder to its purpose, implementation order, coding standards, and integration contracts. Follow this guide top-to-bottom; each layer depends on everything above it.

---

## Table of Contents

1. [Architecture Philosophy](#architecture-philosophy)
2. [Coding Standards & Pattern Reference](#coding-standards--pattern-reference)
3. [Folder Implementation Order](#folder-implementation-order)
4. [Phase 1 — Core Foundation](#phase-1--core-foundation)
5. [Phase 2 — Hardware Abstraction Layer](#phase-2--hardware-abstraction-layer)
6. [Phase 3 — AI System](#phase-3--ai-system)
7. [Phase 4 — Backend Infrastructure](#phase-4--backend-infrastructure)
8. [Phase 5 — Applications](#phase-5--applications)
9. [Phase 6 — Tests, Scripts & Deployment](#phase-6--tests-scripts--deployment)
10. [Cross-Cutting Concerns](#cross-cutting-concerns)
11. [Dependency Graph](#dependency-graph)
12. [Quick-Start Checklist](#quick-start-checklist)

---

## Architecture Philosophy

This project follows a **layered, modular architecture** where no layer imports from a layer above it:

```
┌─────────────────────────────────────────────────┐
│            Applications (Layer 5)               │
│   watch_firmware │ mobile_companion │ desktop    │
├─────────────────────────────────────────────────┤
│            Backend Services (Layer 4)           │
│  api │ services │ storage │ monitoring          │
├─────────────────────────────────────────────────┤
│            AI System (Layer 3)                  │
│  conversational_ai │ health │ personality │ task │
├─────────────────────────────────────────────────┤
│        Hardware Abstraction (Layer 2)           │
│  sensors │ holographic │ power │ communication  │
├─────────────────────────────────────────────────┤
│           Core Foundation (Layer 1)             │
│  constants │ exceptions │ interfaces │ config    │
├─────────────────────────────────────────────────┤
│                  Core Utils                     │
│  math │ data │ validation │ logging │ time │ mem │
└─────────────────────────────────────────────────┘
```

**Rules:**
- Lower layers NEVER import from upper layers
- Each module exposes a clean public API via `__init__.py`
- All public classes carry type annotations
- All I/O-bound operations support `async/await`
- Thread-safety via `threading.Lock` / `threading.RLock`
- Global singleton instances are created at module bottom

---

## Coding Standards & Pattern Reference

Every `.py` file in `src/` follows this exact structure. Study `src/core/utils/data_utils.py` as the gold standard.

### File Template

```python
"""
[Module Title] for AI Holographic Wristwatch System

[2-3 sentence description of what this module does, what problem
it solves, and its role in the overall architecture.]
"""

# ── Standard Library ────────────────────────────────────────────
import ...

# ── Third-Party ─────────────────────────────────────────────────
import numpy as np
import ...

# ── Type Variables ───────────────────────────────────────────────
T = TypeVar('T')

# ── Enums ────────────────────────────────────────────────────────
class FooType(Enum):
    """Short description."""
    VALUE_A = "value_a"

# ── Dataclasses ──────────────────────────────────────────────────
@dataclass
class FooData:
    """Structured data container."""
    field: str
    ...

# ── Protocols / ABCs ─────────────────────────────────────────────
class FooInterface(ABC):
    @abstractmethod
    def do_thing(self) -> Result: ...

# ── Concrete Implementations ─────────────────────────────────────
class FooManager:
    """Full implementation with thread-safety and async support."""
    def __init__(self):
        self._lock = threading.RLock()
    ...

# ── Custom Exceptions ─────────────────────────────────────────────
class FooError(Exception): ...

# ── Global Instances ──────────────────────────────────────────────
global_foo = FooManager()

# ── Convenience Functions ─────────────────────────────────────────
def do_foo_thing(...) -> ...: ...

# ── Tests ─────────────────────────────────────────────────────────
def run_foo_tests(): ...

# ── Module Init ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_foo_tests()
```

### Required Patterns in Every File

| Pattern | Why |
|---------|-----|
| `@dataclass` for all structured data | Immutable-by-default, auto `__repr__` |
| `Enum` for all categorical constants | Prevents magic strings |
| `ABC` / `Protocol` for all interfaces | Enforces contracts, enables mocking |
| `threading.RLock()` in stateful classes | Thread safety for concurrent access |
| `async def` variants for I/O methods | Non-blocking hardware and network calls |
| `@contextmanager` for resource mgmt | Guaranteed cleanup |
| `@functools.wraps` on decorators | Preserves function metadata |
| Global instances at module bottom | Single source of truth per process |
| `run_X_tests()` function | Self-contained smoke test |

---

## Folder Implementation Order

Build in this exact sequence. **Do not skip ahead** — later layers import earlier ones.

```
Step  1  src/core/constants.py
Step  2  src/core/exceptions.py
Step  3  src/core/interfaces/
Step  4  src/core/config/
         └── base_config.py → ai_config.py → device_config.py
             → security_config.py → feature_flags.py
Step  5  src/hardware/sensors/
         └── biometric → environmental → motion_tracking
             → audio → fusion
Step  6  src/hardware/holographic/
         └── projector_control → hologram_generation
             → rendering → display_calibration → interaction_tracking
Step  7  src/hardware/power_management/
Step  8  src/hardware/communication/
Step  9  src/hardware/wristwatch/
Step 10  src/ai_system/conversational_ai/
Step 11  src/ai_system/personality_engine/
Step 12  src/ai_system/health_monitoring/
Step 13  src/ai_system/knowledge_systems/
Step 14  src/ai_system/learning_systems/
Step 15  src/ai_system/task_automation/
Step 16  src/ai_system/security_privacy/
Step 17  src/backend/api/
Step 18  src/backend/services/
Step 19  src/backend/storage/
Step 20  src/backend/security/
Step 21  src/backend/monitoring/
Step 22  src/backend/data_processing/
Step 23  src/backend/ai_infrastructure/
Step 24  src/applications/watch_firmware/
Step 25  src/applications/mobile_companion/
Step 26  src/applications/desktop_interface/
Step 27  src/applications/development_tools/
Step 28  tests/ (unit → integration → e2e → performance → security)
Step 29  scripts/
Step 30  config/
Step 31  deployment/
```

---

## Phase 1 — Core Foundation

> **Must be 100% complete before any other phase begins.**

### `src/core/constants.py`

**Purpose:** Single source of truth for all magic numbers, string keys, and system-wide enumerations that are shared across every module.

**What to implement:**
- Hardware limits (display resolution, frame rates, sensor sampling rates)
- AI system limits (context windows, token budgets, confidence thresholds)
- Network constants (timeouts, retry counts, WebSocket ping intervals)
- Security constants (key lengths, hash iterations, session durations)
- Business logic constants (health thresholds, battery levels, privacy tiers)

**Key classes/enums:**
```python
class SystemConstants         # Version, build, device IDs
class DisplayConstants        # Holographic display specs
class SensorConstants         # Sampling rates, ranges, calibration
class AIConstants             # Model parameters, thresholds
class NetworkConstants        # Timeouts, endpoints, protocols
class SecurityConstants       # Crypto parameters, key sizes
class HealthConstants         # Vital sign normal ranges
class BatteryConstants        # Charge levels, power modes
class PrivacyConstants        # Data retention, consent tiers
```

**Imports from:** nothing (leaf node)
**Imported by:** everything

---

### `src/core/exceptions.py`

**Purpose:** Complete exception hierarchy for the entire system. Every module raises typed exceptions from this hierarchy — never bare `Exception`.

**What to implement:**
- Base `AIWristwatchError` with correlation ID, severity, and recovery hint
- Hardware exceptions (sensor failures, display errors, power faults)
- AI exceptions (model errors, context overflow, safety violations)
- Network/API exceptions (timeout, auth, rate limit)
- Security exceptions (tamper detection, key errors, access denied)
- Data exceptions (corruption, integrity, schema mismatch)
- Application exceptions (UI errors, sync failures)

**Key classes:**
```python
class AIWristwatchError(Exception)      # root
class HardwareError(AIWristwatchError)
  └─ SensorError, DisplayError, PowerError, CommunicationError
class AISystemError(AIWristwatchError)
  └─ ConversationError, ModelError, SafetyViolationError
class SecurityError(AIWristwatchError)
  └─ AuthenticationError, AuthorizationError, TamperDetectionError
class DataError(AIWristwatchError)
  └─ DataCorruptionError, SchemaValidationError, IntegrityError
class NetworkError(AIWristwatchError)
  └─ TimeoutError, ConnectionError, RateLimitError
class ApplicationError(AIWristwatchError)
  └─ SyncError, UIError, ConfigurationError
```

**Imports from:** `constants.py`
**Imported by:** everything

---

### `src/core/interfaces/`

**Purpose:** Abstract contracts (ABCs and Protocols) that every concrete implementation must satisfy. These enable mock-based testing, dependency injection, and clean layering.

**Files and their contracts:**

#### `ai_assistant_interface.py`
```python
class AIAssistantInterface(ABC):
    async def process_input(self, input: UserInput) -> AIResponse
    async def generate_response(self, context: ConversationContext) -> str
    async def get_personality_state(self) -> PersonalityState
    async def update_user_model(self, interaction: Interaction) -> None
    def get_confidence_score(self) -> float
    def is_safe_response(self, response: str) -> bool
```

#### `display_interface.py`
```python
class HolographicDisplayInterface(ABC):
    async def render_hologram(self, hologram: HologramData) -> RenderResult
    async def calibrate(self) -> CalibrationResult
    def set_brightness(self, level: float) -> None
    def get_display_status(self) -> DisplayStatus
    def project_ui_element(self, element: UIElement) -> None
```

#### `sensor_interface.py`
```python
class SensorInterface(ABC):
    async def read(self) -> SensorReading
    async def calibrate(self) -> bool
    def get_sensor_info(self) -> SensorInfo
    def is_healthy(self) -> bool
    def get_sampling_rate(self) -> float

class SensorFusionInterface(ABC):
    async def fuse_readings(self, readings: List[SensorReading]) -> FusedData
```

#### `knowledge_interface.py`
```python
class KnowledgeBaseInterface(ABC):
    async def query(self, query: str, context: QueryContext) -> KnowledgeResult
    async def store(self, fact: Fact) -> str
    async def update(self, fact_id: str, data: Dict) -> bool
    def get_knowledge_stats(self) -> KnowledgeStats
```

#### `task_interface.py`
```python
class TaskExecutorInterface(ABC):
    async def execute(self, task: Task) -> TaskResult
    async def schedule(self, task: Task, trigger: TaskTrigger) -> str
    def cancel(self, task_id: str) -> bool
    def get_task_status(self, task_id: str) -> TaskStatus
```

**Imports from:** `constants.py`, `exceptions.py`
**Imported by:** all implementation layers

---

### `src/core/config/`

**Purpose:** Pydantic-based (or dataclass-based) configuration system with validation, environment overrides, secret management, and hot-reloading.

**Files:**

#### `base_config.py`
- `BaseConfiguration` dataclass with environment detection
- `ConfigLoader` — reads from YAML/JSON/env vars with priority order
- `ConfigValidator` — validates all config fields on startup
- `ConfigWatcher` — hot-reload on file change (uses `watchdog`)
- Global `global_config_loader` instance

#### `ai_config.py`
- `ModelConfig` — model ID, temperature, max tokens, stop sequences
- `PersonalityConfig` — name, tone, adaptation rate, ethical constraints
- `ConversationConfig` — context window, history length, language
- `LearningConfig` — learning rate, memory decay, personalization level
- `SafetyConfig` — content filters, confidence thresholds, fallback behavior

#### `device_config.py`
- `WristwatchConfig` — display specs, sensor list, battery profile
- `HolographicConfig` — projection angle, resolution, brightness limits
- `SensorConfig` — per-sensor sampling rate, calibration offsets
- `PowerConfig` — power modes, charging thresholds, thermal limits
- `CommunicationConfig` — BLE, WiFi, cellular parameters

#### `security_config.py`
- `EncryptionConfig` — algorithm, key size, rotation policy
- `AuthenticationConfig` — biometric settings, fallback methods, lockout
- `PrivacyConfig` — data retention, consent levels, anonymization
- `ComplianceConfig` — regional regulations, audit requirements

#### `feature_flags.py`
- `FeatureFlag` enum of all toggleable features
- `FeatureFlagManager` — enable/disable features at runtime
- A/B testing support with percentage rollouts
- Remote flag override from backend

**Imports from:** `constants.py`, `exceptions.py`
**Imported by:** all other modules

---

## Phase 2 — Hardware Abstraction Layer

> Implements the interfaces defined in Phase 1. Zero business logic — only hardware communication.

### `src/hardware/sensors/`

Each sensor module follows the same pattern: implement `SensorInterface`, add calibration, noise filtering, and async streaming.

#### `biometric/`
- `heart_rate_monitor.py` — PPG signal processing, HRV calculation, arrhythmia detection
- `spo2_monitor.py` — blood oxygen saturation via dual-wavelength LED
- `skin_temperature.py` — continuous skin temp with drift correction
- `galvanic_skin_response.py` — electrodermal activity for stress detection
- `blood_pressure_estimator.py` — cuffless BP estimation via PTT method
- `ecg_processor.py` — single-lead ECG, R-peak detection, QRS analysis

#### `environmental/`
- `air_quality_monitor.py` — VOC, CO2, particulate matter sensors
- `uv_index_sensor.py` — UV-A/B measurement with exposure warnings
- `ambient_light_sensor.py` — lux measurement, color temperature
- `pressure_altitude_sensor.py` — barometric pressure, altitude calculation
- `humidity_temperature.py` — ambient temp/humidity for comfort index

#### `motion_tracking/`
- `accelerometer.py` — 3-axis acceleration, step counting, fall detection
- `gyroscope.py` — angular velocity, orientation tracking
- `magnetometer.py` — compass heading, magnetic interference detection
- `gesture_recognizer.py` — wrist gesture classification via ML

#### `audio/`
- `microphone_array.py` — beamforming, noise cancellation, wake word
- `speaker.py` — audio output, tactile feedback coordination
- `voice_activity_detector.py` — VAD for power-efficient wake detection

#### `fusion/`
- `sensor_fusion_engine.py` — Kalman filter, complementary filter, EKF
- `activity_classifier.py` — ML-based activity recognition from fused data
- `context_inferencer.py` — infer user context from multi-sensor data

**Pattern for each sensor file:**
```python
class HeartRateMonitor(SensorInterface):
    """PPG-based heart rate monitoring with HRV analysis."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self._lock = threading.RLock()
        self._running = False
        self._latest_reading = None
        self._calibration_data = HeartRateCalibration()

    async def read(self) -> SensorReading: ...
    async def calibrate(self) -> bool: ...
    async def stream(self) -> AsyncIterator[SensorReading]: ...
    def is_healthy(self) -> bool: ...
    def get_sensor_info(self) -> SensorInfo: ...
```

---

### `src/hardware/holographic/`

#### `projector_control.py`
- `LaserProjectorController` — safety interlocks, power control, thermal management
- `ProjectionCalibrationData` — geometric distortion correction maps
- `SafetyMonitor` — eye-safety compliance, emergency shutdown

#### `hologram_generation.py`
- `HologramGenerator` — computes holographic interference patterns
- `WavefieldCalculator` — wave optics propagation (ASM method)
- `DepthMapProcessor` — converts depth maps to holographic data
- `HologramOptimizer` — ADMM-based phase retrieval for quality

#### `rendering/`
- `HolographicRenderer` — real-time rendering pipeline
- `SceneGraph` — spatial scene representation
- `LightFieldRenderer` — light field display support
- `UIElementRenderer` — 3D UI widgets in holographic space

#### `display_calibration.py`
- `GeometricCalibration` — lens distortion correction
- `ColorCalibration` — per-wavelength intensity correction
- `EyeTrackingCalibration` — viewer position adaptation

#### `interaction_tracking.py`
- `HandTrackingProcessor` — 3D hand pose estimation
- `GazeTracker` — eye gaze for hologram interaction
- `GestureInterpreter` — maps gestures to UI actions

---

### `src/hardware/power_management/`

#### `battery_optimization/`
- `BatteryManager` — state of charge, state of health, cycle counting
- `ChargeController` — adaptive charging, temperature protection
- `PowerBudgetAllocator` — distributes power budget across subsystems

#### `charging_systems/`
- `WirelessCharger` — Qi protocol implementation, foreign object detection
- `SolarCharger` — MPPT algorithm for ambient light harvesting
- `KineticHarvester` — piezoelectric motion energy recovery

#### `monitoring/`
- `PowerConsumptionMonitor` — per-component power tracking
- `ThermalManager` — temperature monitoring, throttling, ventilation
- `PowerModeController` — idle, low-power, high-performance, emergency modes

---

### `src/hardware/communication/`

#### `wireless/`
- `BluetoothLEManager` — GATT server/client, pairing, bonding
- `WiFiManager` — 802.11ax, WPA3, fast BSS transition
- `NearFieldCommunication` — ISO 14443, NDEF, secure element
- `UltraWideBandRanging` — centimeter-precision spatial positioning

#### `device_pairing/`
- `PairingManager` — secure out-of-band pairing protocol
- `DeviceRegistry` — trusted device store with attestation
- `SessionManager` — encrypted session establishment and key rotation

#### `cloud_connectivity/`
- `CloudClient` — REST/WebSocket client with retry and circuit breaker
- `EdgeComputeManager` — task offloading to edge nodes
- `SyncEngine` — bidirectional data sync with conflict resolution

---

### `src/hardware/wristwatch/`

#### `chassis/`
- `ChassisController` — physical hardware state, tamper detection
- `HapticController` — precision haptic feedback patterns
- `ButtonController` — physical button debouncing and chording

#### `display_system/`
- `WatchfaceEngine` — composite display (OLED + holographic)
- `BrightnessController` — adaptive ambient light response
- `DisplayPowerManager` — always-on display optimization

#### `input_systems/`
- `TouchController` — force-sensitive touchpad processing
- `DigitalCrownController` — rotary encoder with haptic detents
- `VoiceInputController` — on-device wake word + STT pipeline

#### `miniaturization/`
- `ChipsetManager` — multi-die integration, thermal interface
- `MemoryController` — unified memory management across dies

#### `strap_systems/`
- `BiometricStrapController` — extended sensor array in strap
- `StrapConnectionManager` — hot-swap strap detection and init

---

## Phase 3 — AI System

> Business logic and intelligence. Depends on Phase 1 and 2.

### `src/ai_system/conversational_ai/`

#### `natural_language_understanding/`
- `intent_recognition.py` — classify user intent with confidence scoring
- `entity_extraction.py` — named entity recognition for watch context
- `sentiment_analysis.py` — real-time sentiment with emotional nuance
- `language_detection.py` — 50+ language detection and routing
- `context_understanding.py` — multi-turn context tracking
- `command_parsing.py` — structured command parsing for device control

#### `dialogue_management/`
- `conversation_state.py` — FSM-based conversation state machine
- `turn_taking.py` — interruption handling, barge-in detection
- `topic_tracking.py` — multi-topic conversation graph
- `clarification_handling.py` — ambiguity resolution strategies
- `interruption_management.py` — graceful interruption and resumption

#### `response_generation/`
- `response_composer.py` — LLM-powered response generation
- `template_engine.py` — structured response templates for common tasks
- `multi_modal_formatter.py` — format responses for voice / holographic / text
- `language_adapter.py` — style and formality adaptation per user

#### `voice_synthesis/`
- `text_to_speech.py` — on-device neural TTS with SSML support
- `voice_customizer.py` — user voice preference profiles
- `prosody_controller.py` — emotional prosody, emphasis, pacing
- `multilingual_synthesizer.py` — cross-lingual synthesis

---

### `src/ai_system/personality_engine/`

- `personality_core/` — core personality traits, values, behavioral constraints
- `emotional_modeling/` — emotion state machine, empathy modeling
- `adaptation_engine/` — user preference learning, style adaptation
- `social_intelligence/` — social context awareness, cultural sensitivity

---

### `src/ai_system/health_monitoring/`

- `vital_signs/` — real-time vital sign analysis and trend detection
- `medical_assistance/` — emergency detection, medication reminders
- `wellness_coaching/` — activity goals, nutrition, sleep quality
- `mental_health/` — stress detection, mindfulness prompts

---

### `src/ai_system/knowledge_systems/`

- `knowledge_base/` — structured knowledge graph with semantic search
- `personal_knowledge/` — user-specific facts, preferences, history
- `domain_expertise/` — specialized knowledge (medical, legal, technical)
- `real_time_information/` — live data integration (weather, news, calendar)

---

### `src/ai_system/learning_systems/`

- `memory_systems/` — episodic, semantic, procedural memory
- `online_learning/` — continual learning from interactions
- `personalization/` — adaptive UI, response style, proactive suggestions
- `skill_acquisition/` — new capability acquisition via few-shot learning

---

### `src/ai_system/task_automation/`

- `smart_home_control/` — HomeKit, Google Home, Zigbee, Z-Wave
- `personal_productivity/` — calendar, email, reminders, focus modes
- `external_integrations/` — payment, navigation, communication apps
- `workflow_orchestration/` — multi-step task planning and execution

---

### `src/ai_system/security_privacy/`

- `ai_safety/` — content filtering, bias detection, ethical guardrails
- `access_control/` — RBAC for AI capabilities, parental controls
- `data_protection/` — on-device encryption, differential privacy

---

## Phase 4 — Backend Infrastructure

> Cloud services supporting the wristwatch and companion apps.

### `src/backend/api/`
- `rest/v1/` — versioned REST API with OpenAPI spec
- `websocket/` — real-time bidirectional communication
- `middleware/` — auth, rate limiting, logging, CORS
- `validators/` — request/response validation schemas

### `src/backend/services/`
- `ai_orchestration/` — model routing, load balancing, A/B testing
- `device_services/` — device registration, telemetry, OTA updates
- `user_services/` — account management, preferences, sync
- `notification_services/` — push, email, SMS delivery
- `integration_services/` — third-party API adapters

### `src/backend/storage/`
- `cache/` — Redis-compatible distributed cache
- `blob_storage/` — binary asset storage (audio, hologram data)
- `search/` — full-text and semantic search engine

### `src/backend/security/`
- `authentication/` — JWT, OAuth2, biometric verification
- `encryption/` — key management, HSM integration
- `privacy/` — GDPR/CCPA compliance tooling
- `threat_detection/` — anomaly detection, DDoS protection
- `compliance/` — audit trails, regulatory reporting

### `src/backend/monitoring/`
- `system_health/` — infrastructure metrics (CPU, memory, disk)
- `ai_monitoring/` — model performance, drift detection
- `alerting/` — threshold-based alerts, PagerDuty integration
- `user_experience_monitoring/` — latency, error rates, NPS
- `security_monitoring/` — SIEM integration, threat feeds

### `src/backend/data_processing/`
- `behavior_analytics/` — usage pattern analysis
- `conversation_analytics/` — NLP quality metrics
- `device_analytics/` — hardware telemetry aggregation
- `predictive_analytics/` — churn prediction, health forecasting
- `user_insights/` — personalization signal generation

### `src/backend/ai_infrastructure/`
- `inference_services/` — model serving (ONNX, TensorFlow Lite, CoreML)
- `model_management/` — versioning, A/B deployment, rollback
- `training_infrastructure/` — federated learning coordination
- `knowledge_management/` — knowledge graph maintenance

---

## Phase 5 — Applications

### `src/applications/watch_firmware/`
- `core_system/kernel/` — RTOS task scheduler, interrupt handlers
- `core_system/boot_system/` — secure boot, attestation, init sequence
- `interfaces/` — hardware driver bindings for all sensors
- `security/` — secure enclave, key storage, tamper response
- `update_system/` — delta OTA with rollback protection

### `src/applications/mobile_companion/`
- `ios/` — Swift/SwiftUI companion app with watchOS sync
- `android/` — Kotlin/Jetpack Compose companion with Wear OS sync

### `src/applications/desktop_interface/`
- `electron/` — cross-platform management dashboard
- `web_interface/pwa/` — browser-based companion portal

### `src/applications/development_tools/`
- `simulator/` — full software simulation of hardware (no physical device needed)
- `debugging_tools/` — log viewer, sensor injector, AI conversation replay
- `testing_framework/` — hardware-in-the-loop test runner
- `deployment_tools/` — firmware flash, configuration push

---

## Phase 6 — Tests, Scripts & Deployment

### `tests/` Structure

```
tests/
├── unit/           # Per-module, fully mocked, < 1s each
├── integration/    # Cross-module, real implementations
├── end_to_end/     # Full user scenarios, requires simulator
├── performance/    # Load, stress, benchmark tests
├── security/       # Pen testing, vulnerability scanning
└── usability/      # Automated UX testing, accessibility
```

**Testing Targets per Module:**
- 90%+ line coverage for `src/core/`
- 85%+ line coverage for `src/hardware/`
- 80%+ line coverage for `src/ai_system/`
- 75%+ line coverage for `src/backend/`

### `scripts/` Purpose

| Directory | What the scripts do |
|-----------|---------------------|
| `build_scripts/` | Compile firmware, bundle apps, build AI models |
| `deployment_scripts/` | Deploy to device, cloud, or staging |
| `device_scripts/` | Provision devices, push configs, monitor fleet |
| `ai_scripts/` | Train models, evaluate performance, manage knowledge base |
| `data_scripts/` | ETL pipelines, analytics aggregation |
| `monitoring_scripts/` | Health checks, alert routing, report generation |
| `security_scripts/` | Certificate rotation, key management, incident response |
| `maintenance_scripts/` | Database cleanup, cache invalidation, log archiving |
| `testing_scripts/` | CI test runners, coverage reports, hardware test execution |

### `config/` Structure

```
config/
├── base_configurations/        # Default values for all systems
├── ai_configurations/          # Model params, personality profiles
├── device_configurations/      # Per-device hardware settings
├── environment_configurations/ # dev / staging / prod overrides
├── integration_configurations/ # Third-party API credentials (secrets via vault)
└── security_configurations/    # Crypto settings, compliance rules
```

### `deployment/` Structure

```
deployment/
├── environments/
│   ├── development/   # docker-compose, local hot-reload
│   ├── staging/       # Kubernetes manifests, canary config
│   ├── production/    # Blue-green deployment, health gates
│   └── testing/       # Ephemeral CI environments
```

---

## Cross-Cutting Concerns

### Logging
Every module gets a logger via:
```python
from src.core.utils import get_logger
logger = get_logger(__name__)
```
Structured, JSON-formatted, correlated with `trace_id` and `session_id`.

### Error Handling
Never swallow exceptions. Use the typed hierarchy:
```python
try:
    result = hardware.read()
except SensorError as e:
    logger.error("Sensor read failed", exception=e)
    raise  # or handle gracefully
```

### Configuration Access
```python
from src.core.config import get_config
config = get_config()
sampling_rate = config.device.sensors.heart_rate.sampling_rate_hz
```

### Thread Safety
All shared state uses `threading.RLock()`. Async code uses `asyncio.Lock()`. Never mix.

### Privacy by Design
- Biometric data never leaves device without explicit user consent
- All PII is anonymized before reaching backend analytics
- Data retention enforced by `PrivacyConfig.retention_days`

---

## Dependency Graph

```
src/core/utils/          ← no dependencies (built)
src/core/constants.py    ← no dependencies
src/core/exceptions.py   ← constants
src/core/interfaces/     ← constants, exceptions
src/core/config/         ← constants, exceptions, utils
src/hardware/            ← core/*
src/ai_system/           ← core/*, hardware/
src/backend/             ← core/*, ai_system/
src/applications/        ← core/*, hardware/, ai_system/, backend/
tests/                   ← everything (with mocks)
```

---

## Quick-Start Checklist

Before writing any module, verify:

- [ ] All imports resolve (run `python -c "import module"`)
- [ ] `__init__.py` exports the public API cleanly
- [ ] All public methods have type annotations
- [ ] All classes are thread-safe (or documented as not thread-safe)
- [ ] Custom exceptions inherit from the appropriate base in `exceptions.py`
- [ ] Constants use values from `constants.py`, not magic literals
- [ ] A `run_X_tests()` function exists and passes
- [ ] `if __name__ == "__main__":` block demonstrates usage
- [ ] Async methods have both sync and async signatures where applicable
- [ ] No circular imports (verify with `python -c "from src.X import Y"`)

---

*This document is the engineering constitution of the AI Holographic Wristwatch project. Every pull request is measured against it.*
