import re
from pathlib import Path

import httpx

from apt_classifier.config import Settings

ATTACK_STIX_BASE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack"
)


def normalize_attack_version(version: str) -> str:
    value = version.strip()
    if value.lower() == "latest":
        return "latest"
    value = value.removeprefix("v").removeprefix("V")
    if not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise ValueError(f"Invalid ATT&CK version: {version!r}")
    return value


def attack_bundle_filename(version: str) -> str:
    normalized = normalize_attack_version(version)
    suffix = "" if normalized == "latest" else f"-{normalized}"
    return f"enterprise-attack{suffix}.json"


def attack_bundle_url(version: str) -> str:
    return f"{ATTACK_STIX_BASE_URL}/{attack_bundle_filename(version)}"


def download_attack_bundle(
    settings: Settings, *, force: bool = False, timeout: float = 180.0
) -> Path:
    destination = settings.attack_bundle_path
    if destination.is_file() and destination.stat().st_size and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=timeout) as client,
            client.stream("GET", attack_bundle_url(settings.attack_version)) as response,
            temporary.open("wb") as output,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes():
                output.write(chunk)
        if not temporary.stat().st_size:
            raise RuntimeError("Downloaded ATT&CK bundle is empty")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


download_enterprise_attack = download_attack_bundle
