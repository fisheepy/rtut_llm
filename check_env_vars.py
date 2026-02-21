"""Validate required environment variables and (optional) provider connectivity checks.

Default behavior is non-networked so it can be safely used in restricted CI/runtime
environments where outbound access may be blocked.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser


REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BUCKET_NAME",
]


def check_required_env_vars() -> bool:
    ok = True
    print("Checking required environment variables:")
    for key in REQUIRED_VARS:
        value = os.getenv(key)
        if value:
            print(f"  ✅ {key} is set")
        else:
            print(f"  ❌ {key} is missing")
            ok = False
    return ok


def check_aws_connectivity() -> bool:
    print("\nChecking AWS connectivity:")
    try:
        import boto3

        sts = boto3.client(
            "sts",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-2",
        )
        identity = sts.get_caller_identity()
        print(f"  ✅ STS auth succeeded ({identity.get('Arn', 'unknown ARN')})")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-2",
        )
        s3.head_bucket(Bucket=os.getenv("AWS_BUCKET_NAME"))
        print("  ✅ S3 bucket access succeeded")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ AWS connectivity check failed: {type(exc).__name__}: {exc}")
        return False


def check_openai_connectivity() -> bool:
    print("\nChecking OpenAI connectivity:")
    try:
        import openai

        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.models.list()
        print("  ✅ OpenAI API auth succeeded")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ OpenAI connectivity check failed: {type(exc).__name__}: {exc}")
        return False


def parse_args() -> tuple[bool, bool]:
    parser = ArgumentParser(description="Validate runtime environment variables.")
    parser.add_argument(
        "--check-connectivity",
        action="store_true",
        help="Also validate live AWS/OpenAI connectivity (requires outbound network).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when connectivity checks fail. Only applies with --check-connectivity.",
    )
    args = parser.parse_args()
    return args.check_connectivity, args.strict


def main() -> int:
    check_connectivity, strict = parse_args()
    vars_ok = check_required_env_vars()

    if not vars_ok:
        print("\nMissing required variables. Skipping connectivity checks.")
        return 1

    if not check_connectivity:
        print("\n✅ Required env vars are set.")
        print("ℹ️ Skipping live connectivity checks (use --check-connectivity to enable).")
        return 0

    aws_ok = check_aws_connectivity()
    openai_ok = check_openai_connectivity()

    if vars_ok and aws_ok and openai_ok:
        print("\n✅ All env var checks passed.")
        return 0

    print("\n⚠️ Some connectivity checks failed. This can happen in restricted environments (e.g., no outbound network).")
    if strict:
        return 2

    print("ℹ️ Continuing with success because --strict was not specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
