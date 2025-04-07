from typing import Dict, List, Type

from .analysis_video_profile import AnalysisVideoProfile
from .base_profile import BaseProfile
from .zero_shot_profile import ZeroShotProfile
from .zero_shot_video_profile import ZeroShotVideoProfile

_AVAILABLE_PROFILES: Dict[str, Type[BaseProfile]] = {
    "zero_shot": ZeroShotProfile,
    "zero_shot_video": ZeroShotVideoProfile,
    "analysis_video": AnalysisVideoProfile,
}


def get_available_profiles() -> List[str]:
    return list(_AVAILABLE_PROFILES.keys())


def get_profile(profile_name: str) -> BaseProfile:
    return _AVAILABLE_PROFILES[profile_name]()


__all__ = [
    "get_available_profiles",
    "get_profile",
]
