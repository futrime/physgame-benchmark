from benchmark.profiles.base_profile import BaseProfile
from benchmark.profiles.zero_shot_profile import ZeroShotProfile

AVAILABLE_PROFILES = ["zero_shot"]

def get_profile(profile_name: str) -> BaseProfile:
    return {
        "zero_shot": ZeroShotProfile,
    }[profile_name]()
