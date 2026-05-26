import json
import os
import time
from google import genai
from google.genai import types

# Імпортуємо Colab userdata безпечно, щоб код міг працювати і локально
try:
    from google.colab import userdata
    HAS_COLAB = True
except ImportError:
    HAS_COLAB = False

from google.genai.errors import APIError # Універсальний виловлювач помилок нового SDK

def generate_final_submission():
    """
    Основний пайплайн обробки датасету питань за допомогою Gemini API.
    Реалізує пакетну обробку, стійкість до лімітів (Rate Limits) та валідацію виводу.
    """
    
    # 1. Безпечне отримання API-ключа (Секрети Colab або змінні оточення системи)
    api_key = None
    if HAS_COLAB:
        try: api_key = userdata.get("GEMINI_API_KEY") or userdata.get("GOOGLE_API_KEY")
        except Exception: pass
        
    if not api_key:
        # Якщо запуск не в Colab, шукаємо ключ у системних змінних
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Помилка: Не знайдено API-ключ ані в Секретах Colab, ані в змінних оточення!")
        return

    # Ініціалізація клієнта Google GenAI SDK (версія 2025/2026 років)
    client = genai.Client(api_key=api_key)
    model = "gemini-3-flash-preview"

    input_file = "/content/test_questions.jsonl"
    output_file = "submission.csv"

    if not os.path.exists(input_file):
        print(f"Помилка: Файл {input_file} не знайдено.")
        return

    # Системна інструкція, адаптована під умови челенджу Google Cloud та DOU корпус
    system_instruction = """Ти - агент, який орієнтується у великому та реалістичному IT-корпусі з DOU.ua, можеш знаходити релевантні джерела та давати точні короткі відповіді на три atomic claim prompts для кожного питання. Ти здатний до точного пошуку та синтезу відповідей. На основі вхідного запиту маєш:

- Ідентифікувати найрелевантніші першоджерела серед 3 851 документів, відсікаючи нерелевантні гілки дискусій
- На базі знайденого контексту сформувати три відповіді (c1, c2, c3) на уточнювальні підпитання

Для кожного question_id потрібно передбачити:
- source_urls
- c1
- c2
- c3

Файл має містити заголовок і виглядати так:

question_id,source_urls,c1,c2,c3
q_001,https://dou.ua/lenta/articles/java-salary-2023/|https://dou.ua/forums/topic/41200/,SwiftData,2023,15
q_002,https://dou.ua/forums/topic/12345/,Kyiv,2024,7
q_003,https://dou.ua/lenta/articles/example/,Docker,5,Ubuntu

Вимоги:
- question_id має точно збігатися з test-файлом
- source_urls може містити не більше 10 URL
- якщо передаєш кілька URL, розділяй їх символом |
- URL мають походити з наданого DOU corpus
- c1, c2, c3 мають бути короткими канонічними відповідями
- на кожне питання має бути рівно один рядок"""

    print("Читаємо всі питання з файлу...")
    all_questions = []
    with open(input_file, "r", encoding="utf-8") as in_f:
        for line in in_f:
            if line.strip():
                all_questions.append(json.loads(line))

    total_q = len(all_questions)
    print(f"Завантажено питань: {total_q}")

    # Ініціалізуємо підсумковий файл та записуємо обов'язковий CSV заголовок
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("question_id,source_urls,c1,c2,c3\n")

    # Пакуємо питання по 20 штук. 
    # Оптимальний розмір батчу для утримання контексту мислення (Thinking HIGH) без втрати якості відповідей.
    BATCH_SIZE = 20
    processed_counter = 0

    print("Починаємо пакетну обробку...")

    for i in range(0, total_q, BATCH_SIZE):
        batch = all_questions[i:i + BATCH_SIZE]

        # Конструюємо груповий запит, змушуючи модель суворо дотримуватись структури
        user_parts = ["Будь ласка, оброби наступні питання незалежно одне від одного відповідно до системної інструкції. Для кожного питання згенеруй рівно один CSV-рядок без додаткових пояснень.\n"]
        for q in batch:
            user_parts.append(f"ID: {q.get('question_id')}\nПитання: {q.get('question')}\n---")

        user_input = "\n".join(user_parts)

        # Налаштування конфігурації: вмикаємо режим логічного мислення (Reasoning) моделі Gemini 3
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            system_instruction=[types.Part.from_text(text=system_instruction)]
        )

        max_retries = 5
        delay = 15
        response_text = None

        # Цикл повторних спроб (Retry Mechanism) з експоненційним затиханням для захисту від Rate Limits (429)
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                    config=generate_content_config,
                )
                response_text = response.text
                break
            except APIError as e:
                # Відловлюємо перевантаження сервера або вичерпання хвилинних лімітів Google AI Studio
                if attempt == max_retries - 1:
                    print(f"\nКритична помилка API після {max_retries} спроб: {e}")
                    break 
                print(f"\n[Спроба {attempt+1}] Обмеження API або збій сервера. Очікуємо {delay} сек...")
                time.sleep(delay)
                delay *= 2 # Подвоюємо час очікування для кожної наступної спроби
            except Exception as e:
                print(f"\nНепередбачувана помилка: {e}")
                time.sleep(5)

        # Обробка та валідація отриманих рядків
        if response_text:
            lines = response_text.strip().split("\n")
            valid_lines = []

            for line in lines:
                line_str = line.strip()
                # Фільтруємо markdown-теги та випадкові дублі заголовочного рядка від моделі
                if line_str and "question_id,source_urls" not in line_str and not line_str.startswith("```"):
                    # Перевіряємо відповідність згенерованого рядка до поточного ID з нашого батчу
                    if any(q.get("question_id") in line_str for q in batch):
                        valid_lines.append(line_str)

            # Дозаписуємо валідовані рядки у файл, мінімізуючи втрату даних при збоях
            with open(output_file, "a", encoding="utf-8") as out_f:
                for v_line in valid_lines:
                    out_f.write(v_line + "\n")

            processed_counter += len(batch)
            print(f"Прогрес: [{processed_counter}/{total_q}] питань оброблено.")
        else:
            # Страхувальний механізм: якщо батч повністю впав через API, заповнюємо порожніми CSV-заглушками,
            # щоб зберегти цілісність підсумкового файлу для Kaggle Submission
            print(f"\n[Заглушка] Не вдалося отримати відповідь для батчу. Записуємо пусті рядки.")
            with open(output_file, "a", encoding="utf-8") as out_f:
                for q in batch:
                    out_f.write(f"{q.get('question_id')},,,,\n")
            processed_counter += len(batch)

        # Невелика пауза між батчами для стабілізації потоку запитів
        time.sleep(5)

    print(f"\nОбробку завершено. Файл '{output_file}' готовий.")

if __name__ == "__main__":
    generate_final_submission()
