"""
serve/run_demo.py - Boot the live translation web application.

Usage:
    py serve/run_demo.py
    py serve/run_demo.py --port 8080
    py serve/run_demo.py --test
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_api():
    """Lightweight self-test of the API endpoints."""
    print("Testing translation endpoints...")
    from fastapi.testclient import TestClient
    from serve.app import app

    client = TestClient(app)
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[OK] GET /api/health passed:", res.json().get("status"))

    # 2. Metrics
    res = client.get("/api/metrics")
    assert res.status_code == 200, f"Metrics failed: {res.text}"
    print("[OK] GET /api/metrics passed:", res.json().get("metrics"))

    # 3. Translation
    payload = {
        "text": "Report suspected health cases to the nearest facility.",
        "src_lang": "eng_Latn",
        "tgt_lang": "rag_Latn"
    }
    res = client.post("/api/translate", json=payload)
    assert res.status_code == 200, f"Translation failed: {res.text}"
    data = res.json()
    print("[OK] POST /api/translate passed:")
    print("   Input: ", data["source_text"])
    print("   Output:", data["translated_text"])
    print("   Confidence:", data["confidence_label"])
    print("   Latency:", data["latency_ms"], "ms")

    # 4. Feedback
    fb_payload = {
        "source_text": "Wash hands with soap and water.",
        "translated_text": "Otsuye emikhono ni sabuni hamwene n'amatsi.",
        "corrected_text": "Otsuye emikhono ni sabuni hamwene n'amatsi amalafu.",
        "notes": "Test verification"
    }
    res = client.post("/api/feedback", json=fb_payload)
    assert res.status_code == 200, f"Feedback failed: {res.text}"
    print("[OK] POST /api/feedback passed:", res.json())

    # 5. UI index
    res = client.get("/")
    assert res.status_code == 200, f"Index failed: {res.text}"
    print("[OK] GET / (UI index HTML) passed")

    print("\nAll self-tests passed successfully! The web app is fully functional.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run Maragoli NMT Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--test", action="store_true", help="Run automated API self-tests and exit")
    parser.add_argument("--open", action="store_true", help="Automatically open browser")
    args = parser.parse_args()

    if args.test:
        return test_api()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Please run:")
        print(f"    {sys.executable} -m pip install fastapi uvicorn")
        return 1

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  Lulogooli Translate — Live Translation Web Service")
    print("=" * 60)
    print(f"  Local URL: {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    if args.open:
        webbrowser.open(url)

    uvicorn.run("serve.app:app", host=args.host, port=args.port, reload=True, app_dir=str(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
