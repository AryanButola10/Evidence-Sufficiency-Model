"""
scripts/health_check.py

Phase 1 verification script.

Run this after setting up the environment to confirm:
    1. Python version is correct
    2. All Phase 1 dependencies are installed
    3. Config loads from .env
    4. FastAPI server is reachable at /health
    5. Ollama is running and the dev model is available

Usage (from project root, with venv activated):
    python scripts/health_check.py
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# ----------------------------------------------------------------
# Color helpers for terminal output
# ----------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ----------------------------------------------------------------
# Check 1: Python version
# ----------------------------------------------------------------
def check_python() -> bool:
    section("1. Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 11:
        ok(f"Python {version_str} — compatible")
        return True
    else:
        fail(
            f"Python {version_str} — project requires Python 3.11+. "
            f"Install via Homebrew: brew install python@3.11"
        )
        return False


# ----------------------------------------------------------------
# Check 2: Required packages
# ----------------------------------------------------------------
def check_packages() -> bool:
    section("2. Phase 1 Package Imports")

    packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("pydantic_settings", "Config management"),
        ("dotenv", "Environment loading (python-dotenv)"),
        ("httpx", "HTTP client"),
        ("ollama", "Ollama Python client"),
        ("groq", "Groq API client"),
        ("loguru", "Structured logging"),
        ("pytest", "Testing framework"),
    ]

    all_ok = True
    for module, description in packages:
        try:
            __import__(module)
            ok(f"{module} — {description}")
        except ImportError:
            fail(f"{module} not installed — {description}")
            all_ok = False

    return all_ok


# ----------------------------------------------------------------
# Check 3: Config loading
# ----------------------------------------------------------------
def check_config() -> bool:
    section("3. Configuration Loading")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        if env_example.exists():
            warn(
                ".env not found — copy .env.example to .env and fill in values:\n"
                "    cp .env.example .env"
            )
        else:
            fail(".env and .env.example both missing")
        return False

    try:
        from config import settings
        ok(f"Settings loaded — LLM provider: {settings.llm_provider}")
        ok(f"Ollama dev model: {settings.ollama_dev_model}")
        ok(f"Log level: {settings.log_level}")
        ok(f"ESM thresholds: [{settings.esm_lower_threshold}, {settings.esm_upper_threshold}]")
        return True
    except Exception as e:
        fail(f"Failed to load settings: {e}")
        return False


# ----------------------------------------------------------------
# Check 4: FastAPI server
# ----------------------------------------------------------------
def check_api() -> bool:
    section("4. FastAPI Server")

    try:
        import httpx
        response = httpx.get("http://localhost:8000/health", timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            ok(f"FastAPI running — status: {data.get('status')}")
            ok(f"Phase: {data.get('phase')}")
            return True
        else:
            fail(f"FastAPI responded with status {response.status_code}")
            return False
    except Exception:
        warn(
            "FastAPI not running — start it with:\n"
            "    uvicorn api.main:app --reload\n"
            "    (in a separate terminal, with venv activated)"
        )
        return False


# ----------------------------------------------------------------
# Check 5: Ollama
# ----------------------------------------------------------------
def check_ollama() -> bool:
    section("5. Ollama (Local LLM)")

    # Check if Ollama process is running
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            ok(f"Ollama running — {len(models)} model(s) available")

            if models:
                ok(f"Models: {', '.join(models)}")
            else:
                warn(
                    "No models pulled yet. Run:\n"
                    "    ollama pull llama3.2:3b\n"
                    "    (takes ~5-10 minutes, downloads ~2 GB)"
                )
            return True
        else:
            fail(f"Ollama responded with status {response.status_code}")
            return False
    except Exception:
        warn(
            "Ollama not running.\n"
            "    Install: https://ollama.ai/download\n"
            "    Start: ollama serve  (or open the Ollama app)\n"
            "    Pull model: ollama pull llama3.2:3b"
        )
        return False


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> None:
    print(f"\n{BOLD}{'=' * 50}{RESET}")
    print(f"{BOLD}  Reliability-RAG — Phase 1 Health Check{RESET}")
    print(f"{BOLD}{'=' * 50}{RESET}")

    results = {
        "Python 3.11+": check_python(),
        "Phase 1 packages": check_packages(),
        "Configuration": check_config(),
        "FastAPI server": check_api(),       # Requires server to be running
        "Ollama": check_ollama(),             # Requires Ollama to be running
    }

    # Summary
    section("Summary")
    total = len(results)
    passed = sum(results.values())

    for check, status in results.items():
        if status:
            ok(check)
        else:
            fail(check)

    print(f"\n  Passed: {GREEN}{passed}/{total}{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  ✓ Phase 1 setup complete! Ready for Phase 2.{RESET}\n")
    elif passed >= 3:
        print(f"\n{YELLOW}{BOLD}  ⚠ Core setup complete. See warnings above.{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}  ✗ Setup incomplete. Fix failures above.{RESET}\n")


if __name__ == "__main__":
    main()
