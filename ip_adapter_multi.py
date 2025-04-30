# ⇢ ip_adapter_multi.py
import torch
from PIL import Image
from ip_adapter.ip_adapter import IPAdapterPlusXL   # pentru SDXL; pe SD-1.5 funcţionează idem

# ───────── utilitar mic (stack sigur) ──────────
def _safe_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Uneşte tensori şi menţine dtype-ul primului."""
    if len(tensors) == 1:
        return tensors[0].clone()
    return torch.stack([t.to(tensors[0].dtype) for t in tensors])

class IPAdapterMulti(IPAdapterPlusXL):
    """
    Suportă IP-Adapter «multi-reference».
    Exemple:
        refs = [(pil_person, 0.6), (pil_clothes, 0.3), (pil_bg, 0.1)]
        img  = ip_adapter.generate_multi(refs, prompt="some prompt", ...)
    """

    # ───────── internal: obţine embedding pentru o singură referinţă ─────────
    @torch.no_grad()
    def _embed(self, pil_img: Image.Image, weight: float = 1.0) -> torch.Tensor:
        # 1) processor (image_processor sau processor)
        proc = getattr(self, "image_processor", None) or getattr(self, "processor", None)
        if proc is None:
            from transformers import CLIPImageProcessor
            proc = CLIPImageProcessor()

        # 2) obţinem dtype-ul encoder-ului (fp16 la SDXL)
        device = self.device if hasattr(self, "device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        enc_dtype = next(self.image_encoder.parameters()).dtype

        # 3) pre-procesăm imaginea şi convertim la acelaşi dtype
        data = proc(images=pil_img, return_tensors="pt")
        data = {k: v.to(device=device, dtype=enc_dtype) for k, v in data.items()}

        # 4) extragem embedding-ul şi aplicăm ponderea
        vec = self.image_encoder(**data).image_embeds     # [1, tokens, 1280]
        return vec * weight

    # ───────── API public: generate_multi ─────────
    @torch.no_grad()
    def generate_multi(self, refs, *args, **kw):
        """
        refs = list of (PIL.Image, weight) tuples
        """
        embeds = _safe_stack([self._embed(p, w) for p, w in refs]).sum(0)
        if embeds.dim() == 2:                       # <- ADĂUGAT
            embeds = embeds.unsqueeze(0)            #    [1, tokens, dim]
    
        # trim / pad dacă depăşeşti num_tokens
        embeds = embeds[:, :self.num_tokens, :]
    
        kw["image_embeds"] = embeds
        # hack: pasăm o imagine dummy (nu se foloseşte)
        return super().generate(Image.new("RGB", (8, 8)), *args, **kw)
