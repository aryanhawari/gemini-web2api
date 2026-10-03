"""Model catalogue and @think=N resolution.

MODE_CATEGORY enum from the Gemini frontend JS bundle:
1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE.
The mode id is sent in payload field [79]; thinking depth in [17].
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_MODEL = "gemini-3.6-flash"

THINK_SUFFIX_RE = re.compile(r"@think=([0-4])$")


@dataclass
class ModelInfo:
    name: str                    # clean public model name
    mode: int                    # MODE_CATEGORY enum value -> payload field [79]
    think: int                   # thinking depth: 0 deepest .. 4 shallowest
    extra: Optional[dict] = None # sparse payload overrides, e.g. {31: 2, 80: 3}
    description: str = ""


MODELS: Dict[str, dict] = {
    "gemini-3.7-flash": {"mode": 1, "think": 4, "description": "Flash (latest)"},
    "gemini-3.6-flash": {"mode": 1, "think": 4, "description": "Flash"},
    "gemini-3.5-flash": {"mode": 1, "think": 4, "description": "Flash"},
    "gemini-3.5-flash-thinking": {"mode": 2, "think": 0, "description": "Thinking (longest output)"},
    "gemini-3.1-pro": {"mode": 3, "think": 4,
                       "description": "Pro routing (needs Gemini Advanced cookie, else silently Flash)"},
    "gemini-auto": {"mode": 4, "think": 4, "description": "Auto model selection"},
    "gemini-3.5-flash-thinking-lite": {"mode": 5, "think": 0, "description": "Adaptive thinking"},
    "gemini-flash-lite": {"mode": 6, "think": 4, "description": "Fastest"},
    "gemini-3.1-pro-enhanced": {"mode": 3, "think": 4, "extra": {31: 2, 80: 3},
                                "description": "Pro with extra payload routing flags"},
}


def is_known_model(name):
    """True when the (already @think-stripped) name is in the registry."""
    return bool(name) and name.strip() in MODELS


def resolve_model(name: Optional[str]) -> ModelInfo:
    """Resolves a requested model name (with optional @think=N suffix).

    Unknown names silently fall back to DEFAULT_MODEL.
    """
    clean = (name or DEFAULT_MODEL).strip()
    think_override = None
    m = THINK_SUFFIX_RE.search(clean)
    if m:
        think_override = int(m.group(1))
        clean = clean[: m.start()]
    spec = MODELS.get(clean)
    if spec is None:
        clean = DEFAULT_MODEL
        spec = MODELS[clean]
    think = spec["think"] if think_override is None else think_override
    return ModelInfo(
        name=clean,
        mode=spec["mode"],
        think=think,
        extra=spec.get("extra"),
        description=spec.get("description", ""),
    )


def list_model_infos():
    return [
        ModelInfo(name=name, mode=spec["mode"], think=spec["think"],
                  extra=spec.get("extra"), description=spec.get("description", ""))
        for name, spec in MODELS.items()
    ]
