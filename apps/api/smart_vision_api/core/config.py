"""Configuration Management System

Provides a centralized configuration management system for the Smart Vision API.
Handles configuration for all service components including:

Core Components:
- API endpoint configuration
- Model deployment settings
- Hardware acceleration
- Resource paths
- Environment variables

Features:
- Environment-based configuration with override support
- Type validation and enforcement
- Dynamic path resolution
- Hardware capability detection
- Configurable fallback values
- Runtime configuration updates

Example:
    from app.core.config import settings

    # Access configuration values
    version = settings.VERSION
    model_path = settings.equipment_categorization_model_path
    compute_device = settings.equipment_categorization_device
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


def _cuda_available() -> bool:
    """Resolve CUDA availability lazily.

    Importing torch at module import time makes even auth-only API startup pay the
    full ML stack initialization cost. We only need this check when a device
    property is actually accessed.
    """
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False

class Settings(BaseSettings):
    """Configuration management service."""

    # API settings
    API_PREFIX: str = "/api/v1"
    VERSION: str = "1.0.0"

    # Base directory for resolving relative paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent

    # Resource paths from environment variables
    MODEL_DIR: str = "models"  # Default if env var not set
    LOG_DIR: str = "logs"  # Default if env var not set
    PYTHONPATH: str | None = None

    # Logging settings
    LOG_LEVEL: str = "INFO"  # Default log level

    # Model configuration
    FORCE_CPU: bool = False
    EQUIPMENT_CATEGORIZATION_DEVICE: str = "auto"
    EQUIPMENT_CATEGORIZATION_MODEL_PATH: str = "equipment_categorization"
    RETRIEVER_DEVICE: str = "auto"
    MILVUS_URI: str = "tcp://standalone:19530"  # Default Milvus URI
    HYBRID_IMAGE_COLLECTION: str = "qwen3_vl_image_parts"
    HYBRID_TEXT_COLLECTION: str = "bge_m3_text_parts"
    HYBRID_ATTRS_COLLECTION: str = "attrs_parts_v2"
    HYBRID_MODEL_COLLECTION: str = "bge_m3_model_texts"
    HYBRID_CAPTION_COLLECTION: str = "bge_m3_caption_parts"

    # CORS (for front-end access)
    CORS_ORIGINS: str = "*"  # comma-separated origins or "*"
    MAX_IMAGE_BASE64_LENGTH: int = 8_000_000
    WARMUP_TEXT_SEARCH_ON_STARTUP: bool = True

    # Simple token auth (optional)
    AUTH_ENABLED: bool = False
    AUTH_USERNAME: str = "admin"
    AUTH_PASSWORD: str = "admin123"
    AUTH_TOKEN_TTL_SECONDS: int = 60 * 60 * 24  # 24h

    # Equipment and parts categories
    EQUIPMENTS_CATEGORY: List[str] = [
        "Asher", "CMP", "CVD", "ECD", "Etch", "Furnace", "Implant", "Metrology",
        "PVD", "RTP", "Stepper", "Scanner", "Track", "WET", "MoCVD", "Fab Others",
        "Prober", "Handler", "Tester", "ATE ETC", "Packaging", "Dicing Saw",
        "Wire Bonder", "Die Bonder", "Back Grinder", "PKG ETC", "Chip Mounter",
        "Reflow&Soldering", "Inspection", "Auto Inserter", "SMT ETC"
    ]

    PARTS_CATEGORY: List[str] = [
        "PCBs", "RF", "CMP Parts & Consumables", "Motion Control", "Robot",
        "Chuck & Pedestal", "Process Kit", "MFC & LFC", "Gauge", "Valve", "Laser",
        "Pump", "Chiller & Scrubber", "Part Others"
    ]

    OTHERS_CATEGORY: List[str] = [
        "Display", "General Tester", "Microscope", "Plastic Processing",
        "Printer&Dispenser", "Medical", "Other Industry > Other", "PCB Equip",
        "Solar", "LCD", "Wafer", "Others", "Metalworking"
    ]

    def __init__(self, **kwargs):
        """Initialize settings and create required directories."""
        super().__init__(**kwargs)

        self.log_dir_path.mkdir(exist_ok=True)
        self.model_dir_path.mkdir(exist_ok=True)

    @property
    def log_dir_path(self) -> Path:
        """Resolve log directory path.

        Returns absolute path based on LOG_DIR environment variable.
        If LOG_DIR is absolute, uses it directly.
        If relative, resolves from BASE_DIR.
        """
        path = Path(self.LOG_DIR)
        return path if path.is_absolute() else self.BASE_DIR / path

    @property
    def model_dir_path(self) -> Path:
        """Resolve model directory path.

        Returns absolute path based on MODEL_DIR environment variable.
        If MODEL_DIR is absolute, uses it directly.
        If relative, resolves from BASE_DIR.
        """
        path = Path(self.MODEL_DIR)
        return path if path.is_absolute() else self.BASE_DIR / path

    @property
    def equipment_categorization_device(self) -> str:
        """Determine the appropriate compute device for model operations.

        Selection logic:
        1. Check FORCE_CPU override
        2. Evaluate EQUIPMENT_CATEGORIZATION_DEVICE setting
        3. Verify hardware availability

        Returns:
            str: Selected compute device ("cuda", "mps", or "cpu")
        """
        if self.FORCE_CPU:
            return "cpu"

        device_preference = self.EQUIPMENT_CATEGORIZATION_DEVICE.lower()

        if device_preference == "cpu":
            return "cpu"

        if device_preference == "cuda":
            return "cuda" if _cuda_available() else ("mps" if _mps_available() else "cpu")

        # Default behavior for "auto" setting
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    @property
    def equipment_categorization_model_path(self) -> Path:
        """Resolve the model directory path.

        Evaluates the configured model path and falls back to default location
        if not specified.

        Returns:
            Path: Absolute path to model directory
        """
        return self.model_dir_path / self.EQUIPMENT_CATEGORIZATION_MODEL_PATH
    
    @property
    def retriever_device(self) -> str:
        """Determine the appropriate compute device for model operations.

        Selection logic:
        1. Check FORCE_CPU override
        2. Evaluate EQUIPMENT_CATEGORIZATION_DEVICE setting
        3. Verify hardware availability

        Returns:
            str: Selected compute device ("cuda", "mps", or "cpu")
        """
        if self.FORCE_CPU:
            return "cpu"

        device_preference = self.RETRIEVER_DEVICE.lower()

        if device_preference == "cpu":
            return "cpu"

        if device_preference == "cuda":
            return "cuda" if _cuda_available() else ("mps" if _mps_available() else "cpu")

        # Default behavior for "auto" setting
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    @property
    def cors_origins_list(self) -> List[str]:
        value = (self.CORS_ORIGINS or "").strip()
        if not value or value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]


# Initialize global settings
settings = Settings()
