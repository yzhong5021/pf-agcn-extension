"""
esm_embed.py

calls pretrained ESM-1b model to generate embeddings.
"""

from transformers import EsmTokenizer, EsmModel

class ESM_Embed(nn.Module):
    def __init__(self, model_name="facebook/esm1b_t33_650M_UR50S", max_len=1000):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_len = max_len
        self.cls_id  = self.tokenizer.cls_token_id
        self.eos_id  = self.tokenizer.eos_token_id
        self.pad_id  = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def get_esm_embed(self, seqs):
        seqs = [s if len(s) <= self.max_len else s[:self.max_len] for s in seqs]

        enc = self.tokenizer(
            seqs, return_tensors="pt",
            padding=True, truncation=True, add_special_tokens=True,
            return_attention_mask=True
        )
        
        out = self.model(**enc)

        input_ids = enc["input_ids"]
        residue_mask = (input_ids != self.pad_id) & (input_ids != self.cls_id) & (input_ids != self.eos_id)
        return out.last_hidden_state, residue_mask
