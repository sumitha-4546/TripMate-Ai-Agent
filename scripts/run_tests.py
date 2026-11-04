

import os
import sys
import time
from pathlib import Path

# Setup environment for testing
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["APP_ENV"] = "testing"
os.environ["GEMINI_API_KEY"] = "mock_test_key"
os.environ["GEMINI_MODEL"] = "gemini-1.5-flash"

import tests.test_destination_tool as test_rag
import tests.test_weather_tool as test_weather
import tests.test_agent_routing as test_routing
import tests.test_integration as test_integ

TEST_SUITES = [
    (
        "Destination RAG Tool Tests",
        [
            ("test_tokyo_visa_retrieval", lambda: test_rag.test_tokyo_visa_retrieval(None)),
            ("test_bangkok_customs_retrieval", test_rag.test_bangkok_customs_retrieval),
            ("test_barcelona_safety_retrieval", test_rag.test_barcelona_safety_retrieval),
            ("test_unsupported_destination", test_rag.test_unsupported_destination),
        ],
    ),
    (
        "Weather Forecast Tool Tests",
        [
            ("test_valid_destination_seasonal", test_weather.test_valid_destination_seasonal),
            ("test_invalid_destination", test_weather.test_invalid_destination),
            ("test_missing_date_handled_gracefully", test_weather.test_missing_date_handled_gracefully),
            ("test_api_timeout_fallback", test_weather.test_api_timeout_fallback),
            ("test_api_failure_fallback", test_weather.test_api_failure_fallback),
            ("test_forecast_horizon_limitation", test_weather.test_forecast_horizon_limitation),
        ],
    ),
    (
        "LangGraph Agent Routing Tests",
        [
            ("test_visa_question_routes_to_rag_tool", test_routing.test_visa_question_routes_to_rag_tool),
            ("test_weather_question_routes_to_weather_tool", test_routing.test_weather_question_routes_to_weather_tool),
            ("test_packing_question_routes_to_both_tools", test_routing.test_packing_question_routes_to_both_tools),
            ("test_greeting_invokes_no_tools", test_routing.test_greeting_invokes_no_tools),
            ("test_unsupported_flight_booking_invokes_no_tools", test_routing.test_unsupported_flight_booking_invokes_no_tools),
        ],
    ),
    (
        "End-to-End Integration Tests",
        [
            ("test_end_to_end_packing_flow_with_multi_tool_reasoning", test_integ.test_end_to_end_packing_flow_with_multi_tool_reasoning),
        ],
    ),
]


def run_all_tests():
    print("=" * 80)
    print("           TRIPMATE AI TRAVEL ASSISTANT - TEST SUITE")
    print("=" * 80)

    total_passed = 0
    total_failed = 0
    start_total = time.time()

    for suite_name, tests in TEST_SUITES:
        print(f"\n>> {suite_name} ({len(tests)} tests):")
        for test_name, test_fn in tests:
            t0 = time.time()
            try:
                test_fn()
                duration = (time.time() - t0) * 1000
                print(f"  [PASS] {test_name:<55} ({duration:.1f}ms)")
                total_passed += 1
            except Exception as e:
                duration = (time.time() - t0) * 1000
                print(f"  [FAIL] {test_name:<55} ({duration:.1f}ms)")
                print(f"         Error: {e}")
                total_failed += 1

    total_duration = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {total_passed} PASSED, {total_failed} FAILED in {total_duration:.2f}s")
    print("=" * 80)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
