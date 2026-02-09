"""
Script to wrap the trained DistilBERT toxicity detector model into a pickle file.
This allows the model to be loaded easily in the application.
"""
import os
import sys
import pickle
import joblib
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from ml_models.toxicity_detector.wrapper import ToxicityDetectorWrapper


def wrap_toxicity_model(model_dir: str, output_path: str = None):
    """
    Wrap the trained toxicity detector model into a pickle file.
    
    Args:
        model_dir: Path to the trained model directory
        output_path: Path to save the pickled wrapper (default: model_dir/wrapper.pkl)
    """
    print("=" * 60)
    print("Wrapping Toxicity Detector Model")
    print("=" * 60)
    
    # Resolve paths
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if output_path is None:
        output_path = model_dir.parent / "toxicity_detector_wrapper.pkl"
    else:
        output_path = Path(output_path).resolve()
    
    print(f"📂 Model directory: {model_dir}")
    print(f"💾 Output path: {output_path}")
    
    # Load and wrap the model
    print("\n🔄 Loading model...")
    try:
        wrapper = ToxicityDetectorWrapper(str(model_dir))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Test the wrapper
    print("\n🧪 Testing wrapper...")
    test_texts = [
        "This is a helpful response.",
        "I hate you and want you to die."
    ]
    
    for text in test_texts:
        result = wrapper.predict(text)
        print(f"  Text: '{text[:30]}...'")
        print(f"    Toxic: {result['toxicity_detected']}, Score: {result['toxicity_score']:.4f}")
    
    print("✅ Wrapper test passed")
    
    # Save as pickle
    print(f"\n💾 Saving wrapper to {output_path}...")
    try:
        # Use joblib for better compatibility with large models
        joblib.dump(wrapper, output_path, compress=3)
        print(f"✅ Wrapper saved successfully")
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Error saving wrapper: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✅ Model wrapping complete!")
    print("=" * 60)
    print(f"\n📦 Wrapped model saved to: {output_path}")
    print(f"   Use this file with model_loader.load_toxicity_detector_model()")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wrap toxicity detector model into pickle")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for pickled wrapper (default: model_dir/../toxicity_detector_wrapper.pkl)"
    )
    
    args = parser.parse_args()
    
    try:
        wrap_toxicity_model(args.model_dir, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


