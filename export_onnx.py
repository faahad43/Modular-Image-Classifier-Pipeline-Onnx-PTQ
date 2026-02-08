from models import make_model
import argparse
import torch

def export_onnx(args):
    device = "cpu"
    
    model = make_model(
        arch = args.arch,
        num_classes=args.num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    
    #loading the trained weights
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    #dummy input that match the training input
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    
    print(f"Model successfully exported to: {args.output}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output", default="model_fp32.onnx")
    
    args = parser.parse_args()
    export_onnx(args)
    