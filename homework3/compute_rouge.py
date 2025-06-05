import os
import warnings

import torch
from evaluate import load
from torchtext.data import Field
from torchtext.data import BucketIterator

from model import EncoderDecoder
from utils import DEVICE, DATA_DIR, EOS_TOKEN, BOS_TOKEN, make_mask, convert_batch, indices_to_text, ensure_dir_exists, save_to_file, prepare_data


warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")


def compute_rouge(model, word_field, data_iter):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in data_iter:
            try:
                source_input, target_input_ref, source_mask, target_mask_ref = convert_batch(batch)
                
                source_input = source_input[0:1]
                source_mask = source_mask[0:1]
                ref_text = indices_to_text(target_input_ref[0], word_field.vocab)
                
                if not ref_text.strip():
                    continue
                references.append(ref_text)

                encoder_output = model.encoder(source_input, source_mask)
                target_input = torch.tensor([[word_field.vocab.stoi[BOS_TOKEN]]], 
                                            dtype=torch.long).to(DEVICE)
                max_len = target_input_ref.shape[1] // 2

                for _ in range(max_len):
                    _, target_mask = make_mask(source_input, target_input, pad_idx=1)
                    decoder_output = model.decoder(target_input, encoder_output, source_mask, target_mask)
                    logits = decoder_output[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                    target_input = torch.cat((target_input, next_token), dim=1)
                    if next_token.item() == word_field.vocab.stoi[EOS_TOKEN]:
                        break

                pred_text = indices_to_text(target_input[0], word_field.vocab)
                if not pred_text.strip():
                    pred_text = "<пустой summary>"
                predictions.append(pred_text)
            except Exception:
                continue

    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    try:
        rouge = load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    return results


if __name__ == "__main__":
    train_dataset, test_dataset, word_field = prepare_data()


    vocab = torch.load(os.path.join(DATA_DIR, 'pre_vocabulary.pt'), map_location=DEVICE)
    model = EncoderDecoder(source_vocab_size=len(vocab), target_vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'pre_transformer.pth'), map_location=DEVICE))
    model.eval()

    word_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    word_field.vocab = vocab


    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset),
        batch_sizes=(16, 32),
        shuffle=True,
        device=DEVICE,
        sort=False
    )

    rouge_results = compute_rouge(
        model=model,
        word_field=word_field,
        data_iter=test_iter
    )

    output_dir = "results"
    ensure_dir_exists(output_dir)
    output_file = os.path.join(output_dir, 'prepared_embeddings_rouge_metrics.txt')

    if os.path.exists(output_file):
        os.remove(output_file)

    save_to_file(output_file, f"ROUGE-1: {rouge_results['rouge1']:.4f}\n")
    save_to_file(output_file, f"ROUGE-2: {rouge_results['rouge2']:.4f}\n")
    save_to_file(output_file, f"ROUGE-L: {rouge_results['rougeL']:.4f}\n")
    save_to_file(output_file, f"ROUGE-Lsum: {rouge_results['rougeLsum']:.4f}")