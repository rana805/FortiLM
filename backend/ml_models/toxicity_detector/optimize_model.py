"""
Optimize Toxicity Detector Model for Production

This script optimizes the DistilBERT toxicity detector for:
1. Faster inference (quantization, ONNX conversion)
2. Smaller model size
3. Better CPU/GPU performance
4. Batch processing optimization
"""
import sys
import os
from pathlib import Path
import argparse

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


def optimize_with_quantization(model_dir: str, output_dir: str):
    """Optimize model using INT8 quantization (smaller, faster)."""
    print("=" * 60)
    print("Optimizing Model with INT8 Quantization")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import BitsAndBytesConfig
        
        print(f"\n📂 Loading model from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Configure INT8 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        
        print("\n🔄 Applying INT8 quantization...")
        # Note: For inference, we'll use dynamic quantization instead
        # BitsAndBytesConfig is more for training
        
        # Use PyTorch's dynamic quantization for inference
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        print("✅ Quantization complete")
        
        # Save quantized model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving quantized model to: {output_path}")
        model_quantized.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Compare sizes
        original_size = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors") or Path(model_dir).glob("*.bin"))
        quantized_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors") or output_path.glob("*.bin"))
        
        print(f"\n📊 Size Comparison:")
        print(f"   Original: {original_size / (1024*1024):.2f} MB")
        print(f"   Quantized: {quantized_size / (1024*1024):.2f} MB")
        print(f"   Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install torch transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_with_onnx(model_dir: str, output_dir: str):
    """Convert model to ONNX format for faster inference."""
    print("=" * 60)
    print("Optimizing Model with ONNX Conversion")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import onnx
        import onnxruntime as ort
        
        print(f"\n📂 Loading model from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        # Create dummy input
        dummy_text = "This is a test message for ONNX conversion."
        inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        
        print("\n🔄 Converting to ONNX...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_path / "model.onnx"
        
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True,
        )
        
        print(f"✅ ONNX conversion complete: {onnx_path}")
        
        # Test ONNX model
        print("\n🧪 Testing ONNX model...")
        session = ort.InferenceSession(str(onnx_path))
        
        # Prepare inputs
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        print(f"✅ ONNX model test passed")
        print(f"   Output shape: {outputs[0].shape}")
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_path))
        
        # Compare sizes
        original_size = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors") or Path(model_dir).glob("*.bin"))
        onnx_size = onnx_path.stat().st_size
        
        print(f"\n📊 Size Comparison:")
        print(f"   Original: {original_size / (1024*1024):.2f} MB")
        print(f"   ONNX: {onnx_size / (1024*1024):.2f} MB")
        print(f"   Reduction: {(1 - onnx_size/original_size)*100:.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install torch transformers onnx onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_wrapper(model_dir: str):
    """Optimize the wrapper for faster inference."""
    print("=" * 60)
    print("Optimizing Wrapper for Performance")
    print("=" * 60)
    
    try:
        from ml_models.toxicity_detector.wrapper import ToxicityDetectorWrapper
        import time
        
        print(f"\n📂 Loading model from: {model_dir}")
        wrapper = ToxicityDetectorWrapper(model_dir)
        
        # Benchmark
        test_texts = [
            "This is a helpful response.",
            "I hate you and want you to die.",
        ] * 10  # 20 texts total
        
        print(f"\n⏱️  Benchmarking inference speed...")
        print(f"   Testing with {len(test_texts)} texts")
        
        # Single prediction benchmark
        start = time.time()
        for text in test_texts[:5]:
            wrapper.predict(text)
        single_time = time.time() - start
        print(f"   Single predictions: {single_time/5*1000:.2f} ms per text")
        
        # Batch prediction benchmark
        start = time.time()
        results = wrapper.predict_batch(test_texts)
        batch_time = time.time() - start
        print(f"   Batch predictions: {batch_time/len(test_texts)*1000:.2f} ms per text")
        print(f"   Speedup: {single_time/batch_time:.2f}x faster with batching")
        
        print("\n✅ Wrapper optimization complete")
        print("\n💡 Recommendations:")
        print("   - Use predict_batch() for multiple texts")
        print("   - Consider ONNX conversion for 2-3x speedup")
        print("   - Use GPU if available for 5-10x speedup")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_optimized_wrapper(model_dir: str, output_path: str, optimization_type: str = "onnx"):
    """Create an optimized wrapper with the chosen optimization."""
    print("=" * 60)
    print(f"Creating Optimized Wrapper ({optimization_type.upper()})")
    print("=" * 60)
    
    if optimization_type == "onnx":
        # Create ONNX-optimized wrapper
        onnx_dir = Path(output_path).parent / "model_onnx"
        if optimize_with_onnx(model_dir, str(onnx_dir)):
            print("\n✅ Optimized model ready!")
            print(f"   ONNX model: {onnx_dir}")
            print("\n💡 Update wrapper.py to use ONNX runtime for faster inference")
            return True
    elif optimization_type == "quantized":
        # Create quantized wrapper
        quantized_dir = Path(output_path).parent / "model_quantized"
        if optimize_with_quantization(model_dir, str(quantized_dir)):
            print("\n✅ Optimized model ready!")
            print(f"   Quantized model: {quantized_dir}")
            return True
    else:
        print(f"❌ Unknown optimization type: {optimization_type}")
        print("   Available: 'onnx', 'quantized'")
        return False
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Optimize toxicity detector model")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["onnx", "quantized", "benchmark", "all"],
        default="benchmark",
        help="Optimization type: onnx, quantized, benchmark, or all"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for optimized model"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)
    
    if args.type == "benchmark":
        optimize_wrapper(str(model_dir))
    elif args.type == "all":
        print("Running all optimizations...\n")
        optimize_wrapper(str(model_dir))
        print("\n" + "="*60 + "\n")
        if args.output:
            create_optimized_wrapper(str(model_dir), args.output, "onnx")
            print("\n" + "="*60 + "\n")
            create_optimized_wrapper(str(model_dir), args.output, "quantized")
    else:
        if not args.output:
            args.output = str(model_dir.parent / f"model_{args.type}")
        create_optimized_wrapper(str(model_dir), args.output, args.type)


if __name__ == "__main__":
    main()


