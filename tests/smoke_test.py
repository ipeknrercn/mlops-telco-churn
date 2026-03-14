import requests
import time
import sys


def smoke_test_api():
    """Smoke test - verify API is deployed and responding"""
    api_url = "http://localhost:8000"
    max_retries = 5
    retry_delay = 3

    print("============================================================")
    print("SMOKE TEST - Deployment Verification")
    print("============================================================")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}/{max_retries}] Checking API health...")
            response = requests.get(f"{api_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print("Health check PASSED")
                print(f"   Response: {data}")

                if data.get("model_loaded") is True:
                    print(f"Model loaded: {data.get('model_name')}")
                    print(f"Model version: {data.get('model_version')}")
                    print("============================================================")
                    print("SMOKE TEST PASSED - Service is healthy!")
                    print("============================================================")
                    return 0
                else:
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
