from typing import Dict, List, Type

from .analysis_profile import AnalysisProfile
from .base_profile import BaseProfile
from .zero_shot_profile import ZeroShotProfile

_AVAILABLE_PROFILES: Dict[str, Type[BaseProfile]] = {
    "zero_shot": ZeroShotProfile,
    "analysis": AnalysisProfile,
}


def get_available_profiles() -> List[str]:
    return list(_AVAILABLE_PROFILES.keys())


def get_profile(profile_name: str) -> BaseProfile:
    return _AVAILABLE_PROFILES[profile_name]()


__all__ = [
    "BaseProfile",
    "ZeroShotProfile",
    "get_available_profiles",
    "get_profile",
]
