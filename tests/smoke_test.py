import os
import requests
import time
import sys


def smoke_test_api():
    """Smoke test - verify API is deployed and responding."""
    api_url = os.environ.get("SMOKE_TEST_API_URL", "http://localhost:8000")
    max_retries = 5
    retry_delay = 3
    # In CI (e.g. GitHub Actions) there is no MLflow server, so we only check API is up.
    in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() in ("true", "1")

    print("============================================================")
    print("SMOKE TEST - Deployment Verification")
    print("============================================================")
    if in_ci:
        print("(CI mode: passing when API responds; model may be unloaded)")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}/{max_retries}] Checking API health...")
            response = requests.get(f"{api_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print("Health check PASSED")
                print(f"   Response: {data}")

                if data.get("model_loaded") is True:
                    print(f"Model loaded: {data.get('model_version')}")
                    print("============================================================")
                    print("SMOKE TEST PASSED - Service is healthy!")
                    print("============================================================")
                    return 0
                if in_ci:
                    # In CI we only need the API to be up and returning valid health.
                    print("(CI: API is up; model not required in this environment)")
                    print("============================================================")
                    print("SMOKE TEST PASSED - API is responding.")
                    print("============================================================")
                    return 0
                print("⚠️ Model not loaded yet...")
            else:
                print(f"⚠️ Health check returned status {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("⚠️ Connection refused (API not ready yet)")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Error: {e}")

        if attempt < max_retries:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)

    print("============================================================")
    print("SMOKE TEST FAILED - Service not responding")
    print("============================================================")
    return 1


if __name__ == "__main__":
    exit_code = smoke_test_api()
    sys.exit(exit_code)
