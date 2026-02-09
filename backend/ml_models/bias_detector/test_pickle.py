"""
Test the bias detector pickle file to verify it works correctly.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import joblib
import torch

# Patch torch.load to use CPU
original_load = torch.load
if not torch.cuda.is_available():
    def cpu_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        return original_load(*args, **kwargs)
    torch.load = cpu_load

# Find pickle file
model_dir = Path(__file__).parent
pkl_files = list(model_dir.glob("*.pkl"))

if not pkl_files:
    print("❌ No pickle file found in bias_detector directory")
    print(f"   Searched in: {model_dir}")
    exit(1)

pkl_file = pkl_files[0]
print(f"✅ Found pickle file: {pkl_file}")

# Load wrapper directly with joblib
print(f"\nLoading wrapper from {pkl_file}...")
try:
    # Import wrapper class first to register it
    from ml_models.bias_detector.wrapper import BiasDetectorWrapper
    
    # Load with joblib
    wrapper = joblib.load(pkl_file)
    
    # Ensure model is on CPU and device is set correctly
    if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'to'):
        if not torch.cuda.is_available():
            wrapper.model = wrapper.model.cpu()
            wrapper.device = 'cpu'
            # Also ensure tokenizer inputs go to CPU
            if hasattr(wrapper, 'tokenizer'):
                # Tokenizer will use device from wrapper.device
                pass
    
    # Restore original torch.load
    if not torch.cuda.is_available():
        torch.load = original_load
    
    print("✅ Wrapper loaded successfully")
except Exception as e:
    # Restore original torch.load
    if not torch.cuda.is_available():
        torch.load = original_load
    print(f"❌ Error loading wrapper: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test wrapper
print("\n" + "=" * 60)
print("Testing Bias Detector Wrapper")
print("=" * 60)

# Test cases
test_cases = [
    {
        "text": "All women are bad drivers.",
        "expected_bias": True,
        "expected_types": ["gender"]
    },
    {
        "text": "Black people are more likely to commit crimes.",
        "expected_bias": True,
        "expected_types": ["racial"]
    },
    {
        "text": "Muslims are all terrorists.",
        "expected_bias": True,
        "expected_types": ["religious"]
    },
    {
        "text": "The weather is nice today.",
        "expected_bias": False,
        "expected_types": []
    },
    {
        "text": "Women and men should have equal opportunities.",
        "expected_bias": False,
        "expected_types": []
    },
    {
        "text": "All Asians are good at math and all women are emotional.",
        "expected_bias": True,
        "expected_types": ["racial", "gender"]
    }
]

print("\nRunning test cases...\n")
passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    text = test["text"]
    expected_bias = test["expected_bias"]
    expected_types = set(test["expected_types"])
    
    try:
        result = wrapper.predict(text)
        
        bias_detected = result.get("bias_detected", False)
        bias_types = set(result.get("bias_types", []))
        bias_score = result.get("bias_score", 0.0)
        bias_scores = result.get("bias_scores", {})
        
        # Check if bias detection matches
        bias_match = bias_detected == expected_bias
        
        # Check if types match (allow partial match for multi-label)
        if expected_bias:
            types_match = len(expected_types & bias_types) > 0
        else:
            types_match = len(bias_types) == 0
        
        test_passed = bias_match and types_match
        
        if test_passed:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        print(f"{status} Test {i}: '{text[:50]}...'")
        print(f"   Expected bias: {expected_bias}, Got: {bias_detected}")
        print(f"   Expected types: {expected_types}, Got: {bias_types}")
        print(f"   Bias score: {bias_score:.4f}")
        print(f"   Per-type scores: {bias_scores}")
        
        if not test_passed:
            print(f"   ⚠️  Mismatch detected!")
        print()
        
    except Exception as e:
        failed += 1
        print(f"❌ Test {i} failed with error: {e}")
        print()

# Summary
print("=" * 60)
print("Test Summary")
print("=" * 60)
print(f"✅ Passed: {passed}/{len(test_cases)}")
print(f"❌ Failed: {failed}/{len(test_cases)}")

if failed == 0:
    print("\n🎉 All tests passed! Model is working correctly.")
    exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed. Please review the results.")
    exit(1)

