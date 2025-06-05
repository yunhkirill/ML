import os
import warnings

from utils import Summarizer, DATA_DIR
from utils import prepare_data, save_to_file, ensure_dir_exists


warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")


if __name__ == "__main__":
    train_dataset, test_dataset, word_field = prepare_data()

    test_texts = []
    test_references = []
    for example in test_dataset.examples[:5]:
        source_text = ' '.join(example.source) if example.source else ''
        target_text = ' '.join(example.target) if example.target else ''
        if source_text:
            test_texts.append(source_text)
            test_references.append(target_text)
        if len(test_texts) >= 5:
            break

    custom_texts = [
        "Исследователи из MIT объявили о прорыве в области квантовых вычислений. Они разработали новый алгоритм, значительно ускоряющий квантовые расчеты, что может революционизировать криптографию и анализ данных.",
        "Европейский Союз ввел новые правила по ограничению пластиковых отходов. С следующего года одноразовые пластиковые изделия, такие как соломинки и столовые приборы, будут запрещены во всех странах-членах для продвижения устойчивого развития.",
        "В Италии обнаружен древнеримский артефакт - 2000-летняя статуя. Археологи полагают, что она изображает римского императора и проливает свет на историю региона.",
        "Ожидается, что на следующей неделе тропический шторм обрушится на Карибские острова. Власти готовятся к возможным наводнениям и призывают жителей запастись необходимыми припасами.",
        "Технологическая компания выпустила новый шлем виртуальной реальности, предлагающий захватывающие игровые впечатления с продвинутым отслеживанием движений и дисплеями высокого разрешения."
    ]

    summarizer = Summarizer(model_path=os.path.join(DATA_DIR, 'pre_transformer.pth'), vocab_path=os.path.join(DATA_DIR, 'pre_vocabulary.pt'))

    output_dir = "results"
    ensure_dir_exists(output_dir)
    output_file = os.path.join(output_dir, 'summaries.txt')
    
    save_to_file(output_file, f"Результаты суммаризации", 'w')

    save_to_file(output_file, "=== Тестовые примеры ===\n")
    
    test_summaries = summarizer.batch_generate_summaries(test_texts)
    for i, (text, summary, reference) in enumerate(zip(test_texts, test_summaries, test_references), 1):
        result_str = f"\n{i}. Исходный текст: {text}\n   Сгенерированное резюме: {summary}\n   Эталонный заголовок: {reference}\n"
        save_to_file(output_file, result_str)

    save_to_file(output_file, "\n=== Пользовательские примеры ===\n")
    
    custom_summaries = summarizer.batch_generate_summaries(custom_texts)
    for i, (text, summary) in enumerate(zip(custom_texts, custom_summaries), 1):
        result_str = f"\n{i}. Исходный текст: {text}\n   Сгенерированное резюме: {summary}\n"
        save_to_file(output_file, result_str)