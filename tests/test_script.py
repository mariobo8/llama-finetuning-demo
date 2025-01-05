import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainer import LlamaFineTuner

def test_model_structure():
    print("Testing model structure...")
    
    fine_tuner = LlamaFineTuner()
    if fine_tuner.setup():
        model = fine_tuner.model
        
        # Test model structure
        print("\nModel Structure:")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Number of layers: {len(model.layers)}")
        print(f"Embedding dimension: {model.config['dim']}")
        
        # Test if model is on intended device
        print(f"\nModel device: {next(model.parameters()).device}")

if __name__ == "__main__":
    test_model_structure()