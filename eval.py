import torch
from datasets import dataset_loader
from models import make_model
import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate(args):
    os.makedirs("part_1_results", exist_ok=True)

    # Load data
    train_loader, val_loader, num_classes = dataset_loader(
        dataset=args.dataset,
        path=args.path,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        pin_memory=args.pin_memory,
        image_size=args.image_size
    )

    # Initialize model
    model = make_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=False,   # We'll load weights
        freeze_backbone=False
    )

    # Load trained checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {args.arch} on {args.dataset}")
    cm_path = f"part_1_results/cn_{args.arch}_{args.dataset}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")
    

    # Classification report
    target_names = [f"Class_{i}" for i in range(num_classes)]
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    
    # Save summary
    summary_path = "part_1_results/summary.txt"
    with open(summary_path, "a") as f:
        f.write(f"Experiment: {args.arch} | Dataset: {args.dataset}\n\n")
        
        # Class-wise metrics table
        class_metrics = pd.DataFrame(report_dict).transpose().iloc[:num_classes]
        f.write("Class-wise Metrics:\n")
        f.write(class_metrics.to_string())
        f.write("\n\n")

        # Global metrics table
        global_metrics = pd.DataFrame(report_dict).transpose().iloc[num_classes:]
        f.write("Global Metrics:\n")
        f.write(global_metrics.to_string())
        f.write("\n\n")

    print(f"Evaluation summary saved -> {summary_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pin_memory", default=False)
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--path", default="data")
    parser.add_argument("--ckpt", default="best.pt", help="Path to trained checkpoint")
    args = parser.parse_args()

    evaluate(args)
