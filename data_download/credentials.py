"""Unified credential loading for the SnowFLAKES download workflow.

AWS profiles are loaded through boto3/botocore.  Service-specific values such
as CDSE and USGS usernames are stored as additional keys in the same profile.
Environment variables remain a backwards-compatible fallback.
"""

from __future__ import annotations

import os
from typing import Any

import boto3


def profile_values(profile: str) -> dict[str, Any]:
    """Return all values for a boto3 profile, including custom keys."""
    # Do not hide malformed ~/.aws/credentials files: botocore's parse error
    # is actionable, whereas silently falling back produces misleading
    # "credentials are required" errors later in the query workflow.
    session = boto3.Session(profile_name=profile)
    profiles = session._session.full_config.get("profiles", {})  # botocore config
    return {
        str(key).lower(): value
        for key, value in profiles.get(profile, {}).items()
    }


def _value(values: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key.lower())
        if value not in (None, ""):
            return str(value)
    return None


def cdse_credentials() -> tuple[str | None, str | None]:
    """Return CDSE username/password from the ``cdse`` profile."""
    profile = os.getenv("CDSE_AWS_PROFILE", "cdse")
    values = profile_values(profile)
    return (
        _value(values, "cdse_username", "username") or os.getenv("CDSE_USERNAME"),
        _value(values, "cdse_password", "password") or os.getenv("CDSE_PASSWORD"),
    )


def ers_credentials() -> tuple[str | None, str | None]:
    """Return USGS EROS username/token from the ``usgs-landsat`` profile."""
    profile = os.getenv("ERS_AWS_PROFILE", "usgs-landsat")
    values = profile_values(profile)
    return (
        _value(values, "ers_username", "username") or os.getenv("ERS_USERNAME"),
        _value(values, "ers_token", "token") or os.getenv("ERS_TOKEN"),
    )
