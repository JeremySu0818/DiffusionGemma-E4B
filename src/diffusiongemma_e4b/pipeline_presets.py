from __future__ import annotations

import argparse
import json


PRESETS = {
    "gpu": {
        "target_estimated_tokens": 50000000,
        "target_blocks": 200000,
        "max_steps": 200000,
        "batch_size": 1,
        "grad_accum": 32,
        "learning_rate": "2e-4",
        "train_mode_windows": "lora",
        "train_mode_linux": "qlora",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gpu")
    parser.add_argument("--shell", choices=["powershell", "bash", "json"], default="json")
    args = parser.parse_args()

    values = PRESETS[args.preset]
    if args.shell == "json":
        print(json.dumps(values, indent=2))
        return

    if args.shell == "powershell":
        for key, value in values.items():
            env_key = _env_key(key)
            print(f'$env:{env_key}="{value}"')
        print('$env:DG_TRAIN_MODE=$env:DG_TRAIN_MODE_WINDOWS')
        return

    for key, value in values.items():
        env_key = _env_key(key)
        print(f'export {env_key}="{value}"')
    print('export DG_TRAIN_MODE="$DG_TRAIN_MODE_LINUX"')


def _env_key(key: str) -> str:
    return "DG_" + key.upper()


if __name__ == "__main__":
    main()
