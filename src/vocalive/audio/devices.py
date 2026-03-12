from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_DEFAULT_DEVICE_ALIASES = {"default", "system", "system default"}
_EXTERNAL_DEVICE_ALIASES = {
    "external",
    "auto external",
    "current external",
    "headset",
    "headphones",
}
_HEADSET_DEVICE_KEYWORDS = (
    "earbud",
    "earbuds",
    "headphone",
    "headphones",
    "headset",
    "wireless",
)
_LOW_FIDELITY_COMMUNICATION_DEVICE_KEYWORDS = (
    "ag audio",
    "hands free",
    "handsfree",
)
_LOW_FIDELITY_BLUETOOTH_DEVICE_KEYWORDS = (
    "airpods",
    "bluetooth",
)
_EXTERNAL_DEVICE_KEYWORDS = (
    "adapter",
    "dock",
    "external",
    "interface",
    "usb",
    "webcam",
)
_BUILTIN_DEVICE_KEYWORDS = (
    "built in",
    "built-in",
    "internal microphone",
    "macbook air microphone",
    "macbook microphone",
    "macbook pro microphone",
)


@dataclass(frozen=True)
class InputDeviceMatch:
    index: int | None
    name: str
    selection: str

    @property
    def label(self) -> str:
        if self.index is None:
            return self.name
        return f"{self.name} (id={self.index})"


@dataclass(frozen=True)
class _InputDeviceInfo:
    index: int
    name: str
    normalized_name: str


def resolve_input_device(
    sounddevice: Any,
    requested_device: str | int | None,
    prefer_external: bool = True,
) -> InputDeviceMatch:
    devices = list(_iter_input_devices(sounddevice.query_devices()))
    request_kind, request_value = _classify_requested_device(requested_device)

    if request_kind == "index":
        return _as_match(_require_device_by_index(devices, int(request_value)), selection="explicit")
    if request_kind == "name":
        return _as_match(
            _require_device_by_name(devices, str(request_value)),
            selection="explicit",
        )
    if request_kind == "external":
        external_device = _select_external_input_device(devices, minimum_confidence=0)
        if external_device is None:
            raise ValueError(
                "No external input device was found. "
                "Set VOCALIVE_MIC_DEVICE=default or choose a specific device name or id."
            )
        return _as_match(external_device, selection="external")

    default_device = _resolve_default_input_device(sounddevice, devices)
    if request_kind == "default":
        if default_device is None:
            return InputDeviceMatch(index=None, name="system default input", selection="default")
        return _as_match(default_device, selection="default")

    if prefer_external:
        external_device = _select_external_input_device(devices, minimum_confidence=1)
        if external_device is not None and (
            default_device is None or _looks_builtin_input(default_device.normalized_name)
        ):
            return _as_match(external_device, selection="external")

    if default_device is not None:
        return _as_match(default_device, selection="default")

    return InputDeviceMatch(index=None, name="system default input", selection="default")


def _iter_input_devices(raw_devices: Iterable[Mapping[str, Any]]) -> Iterable[_InputDeviceInfo]:
    for index, raw_device in enumerate(raw_devices):
        max_input_channels = int(raw_device.get("max_input_channels", 0) or 0)
        if max_input_channels < 1:
            continue
        raw_index = raw_device.get("index")
        device_index = int(raw_index) if raw_index is not None else index
        name = str(raw_device.get("name") or f"input-device-{index}")
        yield _InputDeviceInfo(
            index=device_index,
            name=name,
            normalized_name=_normalize_device_name(name),
        )


def _classify_requested_device(
    requested_device: str | int | None,
) -> tuple[str, str | int | None]:
    if isinstance(requested_device, int):
        return ("index", requested_device)
    if requested_device is None:
        return ("auto", None)
    normalized_value = requested_device.strip()
    if not normalized_value:
        return ("auto", None)
    if normalized_value.isdigit():
        return ("index", int(normalized_value))
    normalized_alias = _normalize_device_name(normalized_value)
    if normalized_alias in _DEFAULT_DEVICE_ALIASES:
        return ("default", None)
    if normalized_alias in _EXTERNAL_DEVICE_ALIASES:
        return ("external", None)
    return ("name", normalized_value)


def _resolve_default_input_device(
    sounddevice: Any,
    devices: list[_InputDeviceInfo],
) -> _InputDeviceInfo | None:
    default_settings = getattr(getattr(sounddevice, "default", None), "device", None)
    if isinstance(default_settings, (list, tuple)):
        candidate = default_settings[0] if default_settings else None
    else:
        candidate = default_settings
    if candidate is None:
        return None
    if isinstance(candidate, str) and candidate.strip().isdigit():
        candidate = int(candidate)
    if not isinstance(candidate, int) or candidate < 0:
        return None
    return next((device for device in devices if device.index == candidate), None)


def _require_device_by_index(
    devices: list[_InputDeviceInfo],
    device_index: int,
) -> _InputDeviceInfo:
    matched_device = next((device for device in devices if device.index == device_index), None)
    if matched_device is not None:
        return matched_device
    raise ValueError(
        f"Input device id {device_index} was not found. "
        f"Available inputs: {_format_available_devices(devices)}"
    )


def _require_device_by_name(
    devices: list[_InputDeviceInfo],
    requested_name: str,
) -> _InputDeviceInfo:
    normalized_name = _normalize_device_name(requested_name)
    exact_matches = [
        device for device in devices if device.normalized_name == normalized_name
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(
            f"Multiple input devices match {requested_name!r}: "
            f"{_format_available_devices(exact_matches)}"
        )

    partial_matches = [
        device for device in devices if normalized_name in device.normalized_name
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(
            f"Multiple input devices match {requested_name!r}: "
            f"{_format_available_devices(partial_matches)}"
        )

    raise ValueError(
        f"Input device {requested_name!r} was not found. "
        f"Available inputs: {_format_available_devices(devices)}"
    )


def _select_external_input_device(
    devices: list[_InputDeviceInfo],
    minimum_confidence: int,
) -> _InputDeviceInfo | None:
    scored_devices: list[tuple[int, _InputDeviceInfo]] = []
    for device in devices:
        confidence = _external_device_confidence(device.normalized_name)
        if confidence < minimum_confidence:
            continue
        scored_devices.append((confidence, device))
    if not scored_devices:
        return None
    scored_devices.sort(key=lambda item: (-item[0], item[1].index))
    return scored_devices[0][1]


def _external_device_confidence(normalized_name: str) -> int:
    if _looks_builtin_input(normalized_name):
        return -1
    if _looks_low_fidelity_communication_input(normalized_name):
        return 0
    if any(keyword in normalized_name for keyword in _HEADSET_DEVICE_KEYWORDS):
        return 1
    if any(keyword in normalized_name for keyword in _EXTERNAL_DEVICE_KEYWORDS):
        return 2
    if any(keyword in normalized_name for keyword in _LOW_FIDELITY_BLUETOOTH_DEVICE_KEYWORDS):
        return 0
    return 0


def _looks_low_fidelity_communication_input(normalized_name: str) -> bool:
    if any(
        keyword in normalized_name
        for keyword in _LOW_FIDELITY_COMMUNICATION_DEVICE_KEYWORDS
    ):
        return True
    return any(
        keyword in normalized_name
        for keyword in _LOW_FIDELITY_BLUETOOTH_DEVICE_KEYWORDS
    ) and any(keyword in normalized_name for keyword in _HEADSET_DEVICE_KEYWORDS)


def _looks_builtin_input(normalized_name: str) -> bool:
    return any(keyword in normalized_name for keyword in _BUILTIN_DEVICE_KEYWORDS)


def _normalize_device_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


def _format_available_devices(devices: Iterable[_InputDeviceInfo]) -> str:
    labels = [f"{device.name} (id={device.index})" for device in devices]
    return ", ".join(labels) if labels else "none"


def _as_match(device: _InputDeviceInfo, selection: str) -> InputDeviceMatch:
    return InputDeviceMatch(index=device.index, name=device.name, selection=selection)
