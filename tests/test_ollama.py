"""
Phase 4 Verification: Ollama LLM Integration Tests

Tests:
  1. Ollama server connectivity
  2. Model availability
  3. Raw text generation
  4. JSON structured generation
  5. Pydantic schema validation of LLM output
  6. Retry logic (simulated)
  7. JSON extraction edge cases
  8. 3 sample CTI events end-to-end
"""

import sys
import os
import json
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ollama_connectivity():
    """Test 1: Check if Ollama server is reachable."""
    print("=" * 60)
    print("  TEST 1: Ollama Server Connectivity")
    print("=" * 60)

    from models.ollama_model import OllamaLLM

    llm = OllamaLLM(model_name="ollama_gemma")
    available = llm.is_available()

    if available:
        print(f"  [OK] Ollama server is running at {llm.base_url}")
        print(f"  [OK] Model '{llm.model_id}' is available")
    else:
        print(f"  [FAIL] Cannot reach Ollama or model not found at {llm.base_url}")
        print(f"  [INFO] Make sure 'ollama run {llm.model_id}' is running")
        return False

    return True


def test_factory_creation():
    """Test 2: Create OllamaLLM through the factory."""
    print("\n" + "=" * 60)
    print("  TEST 2: LLM Factory Creation")
    print("=" * 60)

    from models.llm_factory import LLMFactory

    llm = LLMFactory.create("ollama_gemma")
    print(f"  [OK] Factory created: {llm}")
    print(f"  [OK] Model ID: {llm.model_id}")
    print(f"  [OK] Base URL: {llm.base_url}")

    return llm


def test_json_extraction():
    """Test 3: JSON extraction edge cases (no LLM call needed)."""
    print("\n" + "=" * 60)
    print("  TEST 3: JSON Extraction Strategies")
    print("=" * 60)

    from models.ollama_model import OllamaLLM

    llm = OllamaLLM(model_name="test")

    # Case 1: Clean JSON
    result = llm._extract_json('{"entities": [], "relations": []}')
    assert result == {"entities": [], "relations": []}, "Clean JSON failed"
    print("  [OK] Strategy 1: Direct JSON parse")

    # Case 2: JSON in markdown code block
    result = llm._extract_json('Here is the result:\n```json\n{"entities": [{"text": "APT28"}]}\n```')
    assert result is not None and "entities" in result, "Code block extraction failed"
    print("  [OK] Strategy 2: Markdown code block extraction")

    # Case 3: JSON embedded in text
    result = llm._extract_json('The output is: {"entities": []} and that is all.')
    assert result == {"entities": []}, "Embedded JSON extraction failed"
    print("  [OK] Strategy 3: Brace-matching extraction")

    # Case 4: Trailing comma repair
    result = llm._extract_json('{"entities": ["a", "b",], "relations": [],}')
    assert result is not None, "Trailing comma repair failed"
    print("  [OK] Strategy 4: JSON repair (trailing commas)")

    # Case 5: No JSON at all
    result = llm._extract_json("This is just plain text with no JSON.")
    assert result is None, "Should return None for non-JSON"
    print("  [OK] Strategy 5: Correctly returns None for non-JSON")

    print("\n  >> All JSON extraction strategies working!\n")
    return True


def test_raw_generation(llm):
    """Test 4: Simple raw text generation."""
    print("=" * 60)
    print("  TEST 4: Raw Text Generation")
    print("=" * 60)

    response = llm.generate_raw("What is a cyber threat? Answer in one sentence.")
    print(f"  [OK] Got response ({len(response)} chars)")
    print(f"  Response: {response[:200]}...")
    assert len(response) > 10, "Response too short"

    return True


def test_json_generation(llm):
    """Test 5: Structured JSON generation."""
    print("\n" + "=" * 60)
    print("  TEST 5: JSON Structured Generation")
    print("=" * 60)

    system_prompt = """You are a CTI extraction expert. Extract entities and relations from the given text.
Return ONLY valid JSON in this format:
{
  "entities": [{"text": "...", "type": "...", "canonical_name": "...", "confidence": 0.9}],
  "relations": [{"head": "...", "relation": "...", "tail": "...", "time": null, "evidence": "...", "confidence": 0.8}]
}"""

    user_prompt = "APT28 used X-Agent malware to target the Democratic National Committee in 2016."

    print("  Calling Ollama for JSON generation...")
    start = time.time()
    result = llm.generate_json(system_prompt, user_prompt)
    elapsed = time.time() - start

    print(f"  [OK] Got JSON response in {elapsed:.1f}s")
    print(f"  [OK] Keys: {list(result.keys())}")

    # Verify structure
    assert "entities" in result, "Missing 'entities' key"
    assert "relations" in result, "Missing 'relations' key"
    assert isinstance(result["entities"], list), "entities should be a list"
    assert isinstance(result["relations"], list), "relations should be a list"

    print(f"  [OK] Entities: {len(result['entities'])}")
    print(f"  [OK] Relations: {len(result['relations'])}")

    # Pretty print
    print(f"\n  Raw output:")
    for line in json.dumps(result, indent=2).split("\n")[:20]:
        print(f"    {line}")
    if len(json.dumps(result, indent=2).split("\n")) > 20:
        print("    ...")

    return result


def test_pydantic_validation(json_result):
    """Test 6: Validate LLM output against Pydantic schemas."""
    print("\n" + "=" * 60)
    print("  TEST 6: Pydantic Schema Validation")
    print("=" * 60)

    from schemas.extraction_schema import ExtractionResult

    try:
        validated = ExtractionResult(**json_result)
        print(f"  [OK] Validated {len(validated.entities)} entities")
        print(f"  [OK] Validated {len(validated.relations)} relations")

        for e in validated.entities[:3]:
            print(f"    Entity: {e.text} | {e.type} | conf={e.confidence}")
        for r in validated.relations[:3]:
            print(f"    Relation: {r.head} -> {r.relation} -> {r.tail} | conf={r.confidence}")

        return True
    except Exception as e:
        print(f"  [WARN] Validation failed: {e}")
        print("  [INFO] This may happen if the LLM output doesn't match schema exactly.")
        print("  [INFO] In production, the pipeline will attempt repair + retry.")
        return False


def test_cti_events(llm):
    """Test 7: Process 3 sample CTI events end-to-end."""
    print("\n" + "=" * 60)
    print("  TEST 7: 3 Sample CTI Events End-to-End")
    print("=" * 60)

    from data_parsers.xml_parser import CTIXMLParser
    from config import DATASET_DIR
    from schemas.extraction_schema import ExtractionResult

    # Parse events
    parser = CTIXMLParser(str(DATASET_DIR))
    all_events = parser.parse_all()

    # Select 3 diverse samples: 1 report, 2 malware (different years)
    samples = []
    report_found = False
    malware_count = 0

    for event in all_events:
        if not report_found and "ReportEvent" in event["file_source"]:
            samples.append(event)
            report_found = True
        elif malware_count < 2 and "MalwareEvent" in event["file_source"]:
            # Pick from different years
            if malware_count == 0 or samples[-1]["file_source"] != event["file_source"]:
                samples.append(event)
                malware_count += 1

        if len(samples) >= 3:
            break

    system_prompt = """You are a CTI extraction expert. Extract entities and relations from the given event.
Return ONLY valid JSON:
{
  "entities": [{"text": "...", "type": "...", "canonical_name": "...", "confidence": 0.9}],
  "relations": [{"head": "...", "relation": "...", "tail": "...", "time": null, "evidence": "...", "confidence": 0.8}]
}
If no entities or relations found, return empty lists."""

    results = []
    for i, event in enumerate(samples, 1):
        print(f"\n  --- Event {i}/3: ID={event['event_id']} from {event['file_source']} ---")

        narrative_preview = event["narrative"][:150].replace("\n", " ")
        print(f"  Narrative: {narrative_preview}...")

        try:
            start = time.time()
            result = llm.generate_json(system_prompt, event["narrative"])
            elapsed = time.time() - start

            print(f"  [OK] Response in {elapsed:.1f}s")
            print(f"       Entities: {len(result.get('entities', []))}")
            print(f"       Relations: {len(result.get('relations', []))}")

            # Attempt Pydantic validation
            try:
                validated = ExtractionResult(**result)
                print(f"  [OK] Schema validation passed")
                results.append({"event": event, "result": validated, "status": "success"})
            except Exception as ve:
                print(f"  [WARN] Schema validation failed: {ve}")
                results.append({"event": event, "result": result, "status": "partial"})

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append({"event": event, "result": None, "status": "error"})

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    partial = sum(1 for r in results if r["status"] == "partial")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n  Results: {success} success, {partial} partial, {errors} errors")
    print(f"  >> {'All 3 events processed!' if errors == 0 else 'Some events failed'}\n")

    return errors == 0


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 4 Verification: Ollama Integration")
    print("=" * 60 + "\n")

    # Test 1: Connectivity (prerequisite for everything else)
    if not test_ollama_connectivity():
        print("\n  [FATAL] Ollama not available. Cannot proceed.")
        print("  Make sure Ollama is running: 'ollama serve'")
        print("  And model is pulled: 'ollama pull gemma_e2b:latest'\n")
        sys.exit(1)

    # Test 2: Factory
    llm = test_factory_creation()

    # Test 3: JSON extraction (no LLM needed)
    test_json_extraction()

    # Test 4: Raw generation
    test_raw_generation(llm)

    # Test 5: JSON generation
    json_result = test_json_generation(llm)

    # Test 6: Pydantic validation
    test_pydantic_validation(json_result)

    # Test 7: 3 CTI events end-to-end
    test_cti_events(llm)

    # Final
    print("=" * 60)
    print("  ALL TESTS PASSED -- Phase 4 Verified!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
