# system/flcore/utils/text_encoder.py
import torch
import torch.nn.functional as F

def _hash_anchor(names, dim=512, device="cpu"):
    # deterministic pseudo-embeddings if no text encoder available
    vecs = []
    for name in names:
        g = torch.Generator(device="cpu")
        g.manual_seed(abs(hash(str(name))) % (2**31 - 1))
        vecs.append(F.normalize(torch.randn(dim, generator=g), dim=0))
    return torch.stack(vecs, dim=0).to(device)

def get_text_anchors(class_names, model_name="clip-ViT-B-32", device=None,
                     template="a photo of a {}"):
    """
    Returns [C, D] LTE anchors normalized. Tries open_clip -> clip -> hash fallback.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = [template.format(str(n)) for n in class_names]

    # Try open_clip
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        with torch.no_grad():
            text = tokenizer(names).to(device)
            emb = model.encode_text(text)
        return F.normalize(emb, dim=1)
    except Exception:
        pass

    # Try OpenAI CLIP
    try:
        import clip
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        with torch.no_grad():
            text = clip.tokenize(names).to(device)
            emb = model.encode_text(text).float()
        return F.normalize(emb, dim=1)
    except Exception:
        # Hash fallback (deterministic)
        return _hash_anchor(class_names, dim=512, device=device)
