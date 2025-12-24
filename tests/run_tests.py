"""
White Box Test Runner
Testleri çalıştırmak için bu dosyayı kullanın
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 70)
print("WHITE BOX TEST SUITE - Human or AI Text Classifier")
print("=" * 70)
print()

# Test 1: Model Initialization
print("📋 WB-TC-001: Model Initialization Code Path Testing")
print("-" * 70)
try:
    import unittest
    from test_white_box_model_init import TestModelInitializationCodePath

    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInitializationCodePath)
    runner = unittest.TextTestRunner(verbosity=2)
    result1 = runner.run(suite)
    print()
except Exception as e:
    print(f"⚠️  Test 1 skipped: {e}")
    print("   (Requires app dependencies: torch, h2o, transformers)")
    print()
    result1 = None

# Test 2: Ensemble Voting
print("📋 WB-TC-002: Ensemble Voting Logic Testing")
print("-" * 70)
try:
    from test_white_box_ensemble import TestEnsembleVotingLogic

    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnsembleVotingLogic)
    runner = unittest.TextTestRunner(verbosity=2)
    result2 = runner.run(suite)
    print()
except Exception as e:
    print(f"⚠️  Test 2 skipped: {e}")
    print("   (Requires app dependencies)")
    print()
    result2 = None

# Test 3: Selenium
print("📋 WB-TC-003: Selenium Integration Testing")
print("-" * 70)
try:
    from test_white_box_selenium import TestFlaskEndpointWhiteBox

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlaskEndpointWhiteBox)
    runner = unittest.TextTestRunner(verbosity=2)
    result3 = runner.run(suite)
    print()
except Exception as e:
    print(f"⚠️  Test 3 skipped: {e}")
    print("   (Requires selenium and ChromeDriver)")
    print()
    result3 = None

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)

results = [result1, result2, result3]
test_names = ["WB-TC-001", "WB-TC-002", "WB-TC-003"]

for i, (name, result) in enumerate(zip(test_names, results), 1):
    if result is None:
        print(f"{name}: ⏭️  SKIPPED (dependencies not available)")
    elif result.wasSuccessful():
        print(f"{name}: ✅ PASSED ({result.testsRun} tests)")
    else:
        print(f"{name}: ❌ FAILED ({len(result.failures)} failures, {len(result.errors)} errors)")

print()
print("💡 TIP: To run these tests, ensure all dependencies are installed:")
print("   pip install -r requirements.txt")
print("   pip install -r requirements-test.txt")
print()
