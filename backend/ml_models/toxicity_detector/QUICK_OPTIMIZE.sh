#!/bin/bash
# Quick optimization script for toxicity detector

echo "============================================================"
echo "Toxicity Detector Model Optimization"
echo "============================================================"
echo ""

MODEL_DIR="ml_models/toxicity_detector/model"
OUTPUT_DIR="ml_models/toxicity_detector/optimized"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "📂 Model directory: $MODEL_DIR"
echo "💾 Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Benchmark current performance
echo "Step 1: Benchmarking current performance..."
python3 ml_models/toxicity_detector/optimize_model.py "$MODEL_DIR" --type benchmark

echo ""
echo "Step 2: Choose optimization:"
echo "  1. ONNX conversion (recommended - 2-3x faster)"
echo "  2. Quantization (75% smaller, 1.5-2x faster)"
echo "  3. Both"
echo "  4. Skip optimization"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Converting to ONNX..."
        python3 ml_models/toxicity_detector/optimize_model.py "$MODEL_DIR" --type onnx --output "$OUTPUT_DIR/onnx"
        ;;
    2)
        echo ""
        echo "Quantizing model..."
        python3 ml_models/toxicity_detector/optimize_model.py "$MODEL_DIR" --type quantized --output "$OUTPUT_DIR/quantized"
        ;;
    3)
        echo ""
        echo "Running all optimizations..."
        python3 ml_models/toxicity_detector/optimize_model.py "$MODEL_DIR" --type all --output "$OUTPUT_DIR"
        ;;
    4)
        echo "Skipping optimization"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Optimization complete!"
echo ""
echo "💡 Next steps:"
echo "  1. Test optimized model performance"
echo "  2. Update wrapper to use optimized model if needed"
echo "  3. Deploy optimized model to production"


