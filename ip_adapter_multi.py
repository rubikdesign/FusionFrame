#  ⇢ ip_adapter_multi.py  (adaugi în proiect)
import torch, torchvision.transforms as T
from PIL import Image
from ip_adapter.ip_adapter import IPAdapterPlusXL        # ← doar varianta SDXL

class IPAdapterMulti(IPAdapterPlusXL):
    """
    Acceptă listă de imagini-referinţă cu ponderi diferite.
    ex: imgs = [(pil_cloth, 0.7), (pil_bg, 0.3)]
    """
    @torch.no_grad()
    def _embed(self, pil, weight=1.0):
        device = self.ip_adapter.device
        proc   = self.ip_adapter.image_processor
        vec    = self.ip_adapter.image_encoder(
                    **proc(images=pil, return_tensors="pt").to(device)
                 ).image_embeds              # [1, tokens, 1280]
        return vec * weight

    @torch.no_grad()
    def generate_multi(self, refs, *args, **kw):
        """
        refs = list of (PIL.Image, weight) tuples
        """
        embeds = torch.stack([self._embed(p, w) for p, w in refs]).sum(0)
        # trim / pad dacă depăşeşti num_tokens
        embeds = embeds[:, :self.num_tokens, :]

        kw["image_embeds"] = embeds
        # hack: pasăm o imagine dummy (nu se foloseşte)
        return super().generate(Image.new("RGB",(8,8)), *args, **kw)
