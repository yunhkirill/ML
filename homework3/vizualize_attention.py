import os
import warnings

import torch
import matplotlib.pyplot as plt

from model import EncoderDecoder
from torchtext.data import Field
from utils import DEVICE, DATA_DIR, EOS_TOKEN, BOS_TOKEN, prepare_data, make_mask, indices_to_text


warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")


def visualize_attention(model, word_field, source_text, target_text, index, output_dir="results/attention_vizualization"):
    model.eval()
    example_dir = os.path.join(output_dir, f"example_{index}")
    os.makedirs(example_dir, exist_ok=True)

    with torch.no_grad():
        source_tokens = word_field.tokenize(source_text.lower())
        target_tokens = word_field.tokenize(target_text.lower())

        source_indices = [word_field.vocab.stoi[token] for token in source_tokens]
        target_indices = [word_field.vocab.stoi[BOS_TOKEN]] + \
                         [word_field.vocab.stoi[token] for token in target_tokens] + \
                         [word_field.vocab.stoi[EOS_TOKEN]]

        source_input = torch.tensor([source_indices], dtype=torch.long).to(DEVICE)
        target_input_ref = torch.tensor([target_indices], dtype=torch.long).to(DEVICE)

        source_mask, target_mask_ref = make_mask(source_input, target_input_ref, pad_idx=word_field.vocab.stoi['<pad>'])

        source_words = indices_to_text(source_input[0], word_field.vocab).split()
        target_words = indices_to_text(target_input_ref[0], word_field.vocab).split()

        max_words = 20
        source_words_display = source_words[:max_words]
        target_words_display = target_words[:max_words]

        # Энкодер
        encoder_output = model.encoder(source_input, source_mask)
        num_layers = len(model.encoder._blocks)
        num_heads = model.encoder._blocks[0]._self_attn._heads_count
        for layer_idx in range(num_layers):
            layer_dir = os.path.join(example_dir, f"encoder_layer_{layer_idx+1}")
            os.makedirs(layer_dir, exist_ok=True)

            # Self-attention энкодера
            attn_probs = model.encoder._blocks[layer_idx]._self_attn._attn_probs
            for head_idx in range(num_heads):
                plt.figure(figsize=(12, 10))

                attn_matrix = attn_probs[0, head_idx].data.cpu().numpy()[:max_words, :max_words]

                plt.imshow(attn_matrix, cmap='Reds', vmin=0, vmax=1)
                plt.title(f"Encoder Layer {layer_idx+1}, Head {head_idx+1}")

                plt.xticks(range(len(source_words_display)), source_words_display, rotation=45, fontsize=10, ha='right')
                plt.yticks(range(len(source_words_display)), source_words_display, fontsize=10)

                plt.colorbar(label='Attention Weight')
                filename = os.path.join(layer_dir, f"head_{head_idx+1}.png")
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()


        # Декодер
        decoder_output = model.decoder(target_input_ref[:, :-1], encoder_output, source_mask, target_mask_ref[:, :-1, :-1])
        num_layers_decoder = len(model.decoder._blocks)
        num_heads_decoder = model.decoder._blocks[0]._self_attn._heads_count
        for layer_idx in range(num_layers_decoder):
            layer_dir = os.path.join(example_dir, f"decoder_layer_{layer_idx+1}")
            os.makedirs(layer_dir, exist_ok=True)

            # Self-attention декодера
            self_attn_probs = model.decoder._blocks[layer_idx]._self_attn._attn_probs
            for head_idx in range(num_heads_decoder):
                plt.figure(figsize=(12, 10))
                attn_matrix = self_attn_probs[0, head_idx].data.cpu().numpy()[:max_words, :max_words]

                plt.imshow(attn_matrix, cmap='Reds', vmin=0, vmax=1)
                plt.title(f"Decoder Self-Attention Layer {layer_idx+1}, Head {head_idx+1}")

                plt.xticks(range(len(target_words_display)), target_words_display, rotation=45, fontsize=10, ha='right')
                plt.yticks(range(len(target_words_display)), target_words_display, fontsize=10)

                plt.colorbar(label='Attention Weight')
                filename = os.path.join(layer_dir, f"self_head_{head_idx+1}.png")
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()


            # Cross-attention декодера
            cross_attn_probs = model.decoder._blocks[layer_idx]._encoder_attn._attn_probs
            for head_idx in range(num_heads_decoder):
                plt.figure(figsize=(12, 10))
                attn_matrix = cross_attn_probs[0, head_idx].data.cpu().numpy()[:max_words, :max_words]

                plt.imshow(attn_matrix, cmap='Reds', vmin=0, vmax=1)
                plt.title(f"Decoder Cross-Attention Layer {layer_idx+1}, Head {head_idx+1}")

                plt.xticks(range(len(source_words_display)), source_words_display, rotation=45, fontsize=10, ha='right')
                plt.yticks(range(len(target_words_display)), target_words_display, fontsize=10)

                plt.colorbar(label='Attention Weight')
                filename = os.path.join(layer_dir, f"cross_head_{head_idx+1}.png")
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()


if __name__ == "__main__":    
    output_dir = "results/attention_vizualization"
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset, test_dataset, word_field = prepare_data()
    
    vocab = torch.load(os.path.join(DATA_DIR, 'pre_vocabulary.pt'), map_location=DEVICE)
    model = EncoderDecoder(source_vocab_size=len(vocab), target_vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'pre_transformer.pth'), map_location=DEVICE))
    model.eval()

    word_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    word_field.vocab = vocab

    examples = [
        {
            "source": "В Тихом океане ученые обнаружили новый вид глубоководной рыбы с яркой окраской. Рыба была найдена на глубине более 2000 метров и обладает уникальной биолюминесценцией.",
            "target": "Новый вид яркой рыбы найден в Тихом океане"
        },
        {
            "source": "Япония объявила о выделении 10 миллиардов иен на развитие возобновляемых источников энергии. Проект направлен на снижение выбросов углерода к 2030 году.",
            "target": "Япония инвестирует в возобновляемую энергию"
        },
        {
            "source": "Астрономы предсказывают редкий метеорный дождь, который будет виден в северном полушарии в следующую пятницу. Ожидается до 100 метеоров в час.",
            "target": "Редкий метеорный дождь ожидается в пятницу"
        }
    ]

    for i, example in enumerate(examples):
        visualize_attention(model, word_field, example["source"], example["target"], i+1, output_dir)