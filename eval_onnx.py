import torch
import onnxruntime as ort
import numpy as np
from datasets import dataset_loader
from models import make_model
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd


def evaluate_pytorch_model(model, dataloader, device):
    """Evaluate PyTorch model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, macro_f1, weighted_f1


def evaluate_onnx_model(onnx_path, dataloader, input_name):
    """Evaluate ONNX model"""
    session = ort.InferenceSession(onnx_path)
    
    all_preds = []
    all_labels = []
    
    for x, y in dataloader:
        # Convert to numpy
        x_np = x.numpy()
        
        # Run inference
        outputs = session.run(None, {input_name: x_np})
        logits = outputs[0]
        
        preds = np.argmax(logits, axis=1)
        all_preds.append(preds)
        all_labels.append(y.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, macro_f1, weighted_f1


def evaluate_all_models(args):
    os.makedirs("part_2_results", exist_ok=True)
    
    # Load validation dataset
    _, val_loader, num_classes = dataset_loader(
        dataset=args.dataset,
        path=args.path,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        pin_memory=args.pin_memory,
        image_size=args.image_size
    )
    
    results = {}
    
    # 1. Evaluate PyTorch FP32 model
    print("Evaluating PyTorch FP32 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = make_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    acc, macro_f1, weighted_f1 = evaluate_pytorch_model(model, val_loader, device)
    results["PyTorch FP32"] = {
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    }
    print(f"  Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    
    # 2. Evaluate ONNX FP32 model
    if os.path.exists(args.onnx_fp32):
        print("\nEvaluating ONNX FP32 model...")
        acc, macro_f1, weighted_f1 = evaluate_onnx_model(args.onnx_fp32, val_loader, "input")
        results["ONNX FP32"] = {
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        }
        print(f"  Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    else:
        print(f"\nWarning: ONNX FP32 model not found at {args.onnx_fp32}")
    
    # 3. Evaluate ONNX Dynamic INT8 model
    if os.path.exists(args.onnx_dynamic):
        print("\nEvaluating ONNX INT8 Dynamic model...")
        acc, macro_f1, weighted_f1 = evaluate_onnx_model(args.onnx_dynamic, val_loader, "input")
        results["ONNX INT8 Dynamic"] = {
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        }
        print(f"  Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    else:
        print(f"\nWarning: ONNX Dynamic INT8 model not found at {args.onnx_dynamic}")
    
    # 4. Evaluate ONNX Static INT8 model
    if os.path.exists(args.onnx_static):
        print("\nEvaluating ONNX INT8 Static model...")
        acc, macro_f1, weighted_f1 = evaluate_onnx_model(args.onnx_static, val_loader, "input")
        results["ONNX INT8 Static"] = {
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        }
        print(f"  Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    else:
        print(f"\nWarning: ONNX Static INT8 model not found at {args.onnx_static}")
    
    # Save comparison table
    summary_path = "part_2_results/summary.txt"
    df = pd.DataFrame(results).T
    
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"Part 2: ONNX Model Comparison\n")
        f.write(f"Model Architecture: {args.arch}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write("="*80 + "\n\n")
        f.write("Comparison Table:\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        # Calculate accuracy drop
        if "PyTorch FP32" in results:
            pytorch_acc = results["PyTorch FP32"]["Accuracy"]
            f.write("Accuracy Drop from PyTorch FP32:\n")
            for model_name, metrics in results.items():
                if model_name != "PyTorch FP32":
                    drop = (pytorch_acc - metrics["Accuracy"]) * 100
                    f.write(f"  {model_name}: {drop:.2f}%\n")
    
    print(f"\n{'='*80}")
    print(f"Comparison summary saved to: {summary_path}")
    print(f"{'='*80}")
    print("\nComparison Table:")
    print(df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="convnext_tiny", help="Model architecture")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--path", default="data", help="Dataset path")
    
    # Model paths
    parser.add_argument("--ckpt", default="best.pt", help="PyTorch checkpoint path")
    parser.add_argument("--onnx_fp32", default="model_fp32.onnx", help="ONNX FP32 model path")
    parser.add_argument("--onnx_dynamic", default="model_dynamic_quant.onnx", help="ONNX dynamic INT8 model path")
    parser.add_argument("--onnx_static", default="static_model_quant.onnx", help="ONNX static INT8 model path")
    
    args = parser.parse_args()
    
    evaluate_all_models(args)
