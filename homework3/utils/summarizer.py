import os
import typing

import torch
import torchtext
import torch.nn.functional as F
from torchtext.data import Field
from torch.serialization import add_safe_globals

from model import EncoderDecoder
from utils import DEVICE, DATA_DIR, BOS_TOKEN, EOS_TOKEN, text_to_indices, indices_to_text, make_mask


add_safe_globals([torchtext.vocab.Vocab])


class Summarizer:    
    def __init__(
        self,
        model_path: str = os.path.join(DATA_DIR, 'transformer.pth'),
        vocab_path: str = os.path.join(DATA_DIR, 'vocabulary.pt'),
        max_summary_len: int = 20,
        device: torch.device = DEVICE
    ):
        self.device = device
        self.max_summary_len = max_summary_len
        
        self.word_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
        self.word_field.vocab = torch.load(vocab_path, map_location=self.device)
        
        self.model = EncoderDecoder(
            source_vocab_size=len(self.word_field.vocab),
            target_vocab_size=len(self.word_field.vocab)
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.pad_idx = self.word_field.vocab.stoi['<pad>']
        self.bos_idx = self.word_field.vocab.stoi[BOS_TOKEN]
        self.eos_idx = self.word_field.vocab.stoi.get(EOS_TOKEN, 2)

    def generate_summary(self, text: str, max_summary_len: typing.Optional[int] = None) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        max_len = max_summary_len or self.max_summary_len
        
        tokens = self.word_field.preprocess(text)
        if not tokens:
            raise ValueError("No valid tokens after preprocessing")
        
        input_ids = text_to_indices(tokens, self.word_field.vocab).to(self.device)  # [1, seq_len]
        src_mask, _ = make_mask(input_ids, input_ids, self.pad_idx)  # [1, 1, seq_len]
        
        output_ids = torch.tensor([[self.bos_idx]], dtype=torch.long).to(self.device)  # [1, 1]
        
        with torch.no_grad():
            for _ in range(max_len):
                _, tgt_mask = make_mask(input_ids, output_ids, self.pad_idx)
                logits = self.model(input_ids, output_ids, src_mask, tgt_mask)
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1)
                output_ids = torch.cat([output_ids, next_token.unsqueeze(1)], dim=1)
                if next_token.item() == self.eos_idx:
                    break
        
        summary = indices_to_text(output_ids[0], self.word_field.vocab)
        return summary.strip() or "<No summary generated>"

    def batch_generate_summaries(self, texts: typing.List[str], max_summary_len: typing.Optional[int] = None) -> typing.List[str]:
        """Generate summaries for a batch of texts."""
        summaries = []
        for text in texts:
            try:
                summary = self.generate_summary(text, max_summary_len)
                summaries.append(summary)
            except ValueError as e:
                summaries.append(f"Error: {str(e)}")
        return summaries