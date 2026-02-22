"""Quick production smoke test for the deployed FAISS app.

Usage:
  python scripts/validate_heroku.py --url https://your-app.herokuapp.com --question "How do I request leave?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


def call_json(method: str, url: str, **kwargs: Any) -> tuple[int | None, Any]:
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {type(exc).__name__}: {exc}"

    try:
        data = response.json()
    except Exception:
        data = response.text
    return response.status_code, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Heroku deployment endpoints.")
    parser.add_argument("--url", required=True, help="Base URL, e.g. https://rtut-app...herokuapp.com")
    parser.add_argument(
        "--question",
        default="How do I request leave?",
        help="Basic question used for /chat validation.",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("1) GET /status")
    status_code, status_payload = call_json("GET", f"{base_url}/status")
    print(f"   status={status_code}, payload={status_payload}")

    print("2) POST /chat")
    chat_code, chat_payload = call_json(
        "POST",
        f"{base_url}/chat",
        json={"question": args.question, "style": "friendly"},
    )
    print(f"   status={chat_code}, payload={chat_payload}")

    if status_code == 200 and chat_code == 200:
        print("\n✅ Heroku smoke test passed.")
        return 0

    if status_code is None or chat_code is None:
        print("\n⚠️ Heroku smoke test could not reach the service from this environment.")
        return 2

    print("\n❌ Heroku smoke test failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
