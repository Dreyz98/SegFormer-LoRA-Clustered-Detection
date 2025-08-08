import argparse, cv2, torch, numpy as np, yaml
from src.segformer_lora import SegFormerWithLoRA

def load_image(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    x = torch.from_numpy(img_resized).float().permute(2,0,1)/255.0
    return img, x.unsqueeze(0)

def colorize(mask):
    # simple color map for classes 0..2
    palette = np.array([[0,0,0],[255,0,0],[0,255,0]], dtype=np.uint8)
    colored = palette[mask]
    return colored

def main(args):
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg.get("device","cpu"))
    model = SegFormerWithLoRA(num_classes=cfg["num_classes"], img_size=cfg["img_size"]).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()
    orig, x = load_image(args.image, cfg["img_size"])
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        mask = logits.argmax(1)[0].cpu().numpy()
    overlay = (0.6*orig + 0.4*colorize(cv2.resize(mask, (orig.shape[1], orig.shape[0])))).astype(np.uint8)
    out = args.output or "assets/pred.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out}")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", default="models/segformer_lora.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="assets/pred.png")
    args = parser.parse_args()
    main(args)
