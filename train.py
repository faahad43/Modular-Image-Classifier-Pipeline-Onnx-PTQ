import torch
from datasets import dataset_loader
from models import make_model
import argparse
import random
import numpy as np

def train(args):
    
    train_dataloader, val_dataloader, num_classes = dataset_loader(
        dataset=args.dataset,
        path= args.path,
        num_worker= args.num_worker,
        batch_size= args.batch_size,
        pin_memory= args.pin_memory,
        image_size= args.image_size
    )
    
    model = make_model(
        arch = args.arch,
        num_classes= num_classes,
        pretrained= args.pretrained,
        freeze_backbone= args.freeze_backbone
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    epochs = args.epochs
    best_val_acc = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Conditional weight decay
    if args.use_weight_decay:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Conditional scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # GradScaler for AMP
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    def accuracy(logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()
    
    for epoch in range(epochs):
        #Train model
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.use_grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                
                # Gradient clipping
                if args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            tr_loss += loss.item()
            tr_acc += accuracy(logits.detach(), y)
            
        tr_loss /= len(train_dataloader)
        tr_acc /= len(train_dataloader)
        
        if scheduler is not None:
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                val_acc += accuracy(logits, y)
                
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        
        
        print(f"Epoch {epoch+1}/{epochs} | "
            f"train loss: {tr_loss:.4f}, Accuracy: {tr_acc:.4f} | "
            f"validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        #Checkpointing to save the best accuracy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, "best.pt")
            print("saved -> best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone layers")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for dataloader")
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--path", default="data")
    
    # Training toggles
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--use_grad_clip", action="store_true", help="Use gradient clipping")
    parser.add_argument("--use_weight_decay", action="store_true", help="Use weight decay in optimizer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train(args)