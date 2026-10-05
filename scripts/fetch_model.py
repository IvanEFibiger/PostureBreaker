from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

_CDN = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

MODELS: dict[str, dict[str, object]] = {
    "heavy": {
        "file": "pose_landmarker_heavy.task",
        "url": f"{_CDN}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "sha256": "64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B",
        "size": 30664242,
    },
    "full": {
        "file": "pose_landmarker_full.task",
        "url": f"{_CDN}/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "sha256": "5134A3AAD27A58B93DA0088D431F366DA362B44E3CCFBE3462B3827A839011B1",
        "size": 9398198,
    },
    "lite": {
        "file": "pose_landmarker_lite.task",
        "url": f"{_CDN}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "sha256": "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A",
        "size": 5777746,
    },
}

DEFAULT_VARIANT = "heavy"
_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def describe_variants() -> str:
    lines = []
    for name, spec in MODELS.items():
        megabytes = int(spec["size"]) / 1_048_576
        lines.append(f"  {name:<6} {megabytes:5.1f} MB")
    return "\n".join(lines)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(url) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def fetch_model(variant: str, dest: Path, force: bool = False) -> Path:
    if variant not in MODELS:
        raise ValueError(f"Variante desconocida: {variant}. Opciones: {', '.join(MODELS)}.")
    expected_sha = str(MODELS[variant]["sha256"])

    if dest.exists() and not force:
        if sha256_file(dest) == expected_sha:
            return dest
        print(f"{dest} existe pero el hash no coincide; lo vuelvo a descargar.")

    _download(str(MODELS[variant]["url"]), dest)

    actual_sha = sha256_file(dest)
    if actual_sha != expected_sha:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"El modelo descargado no pasó la verificación de integridad: "
            f"{actual_sha} != {expected_sha}."
        )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga y verifica el modelo de MediaPipe Pose.")
    parser.add_argument("--variant", choices=sorted(MODELS), default=DEFAULT_VARIANT)
    parser.add_argument("--dest", type=Path, default=Path("pose_landmarker.task"))
    parser.add_argument("--force", action="store_true", help="Ignora un archivo existente.")
    args = parser.parse_args(argv)

    try:
        path = fetch_model(args.variant, args.dest, force=args.force)
    except (ValueError, OSError, URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Modelo listo en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
