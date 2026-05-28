# Порівняльний аналіз стратегій пам'яті для діалогових систем

Курсова робота з дисципліни «Технології машинного навчання та штучний інтелект»  
Державний університет «Житомирська політехніка», 2026
Студентка групи ІПЗм-25-2 Поліщук Карина

## Структура проєкту

```
memory_chatbot/
├── memories/
│   ├── __init__.py          # пакет стратегій
│   ├── base.py              # абстрактний базовий клас
│   ├── buffer_memory.py     # стратегія 1: ковзне вікно
│   ├── summary_memory.py    # стратегія 2: LLM-стискання
│   └── vector_memory.py     # стратегія 3: семантичний пошук
├── chatbot.py               # діалоговий агент (патерн Strategy)
├── evaluation.py            # метрики та тестові сценарії
├── experiment.py            # головний скрипт експерименту
├── visualize.py             # побудова графіків для записки
├── test_day1.py             # тести BufferMemory + Chatbot
├── test_day2.py             # тести SummaryMemory
├── test_day3.py             # тести VectorMemory
├── requirements.txt
├── .env.example
└── README.md
```

## Встановлення

```bash
# 1. Клонувати або розпакувати проєкт
cd memory_chatbot

# 2. Встановити залежності
pip install -r requirements.txt

# 3. Створити файл з API ключем
cp .env.example .env
# Відкрити .env і вставити свій GEMINI_API_KEY
```

Отримати безкоштовний Gemini API ключ: https://aistudio.google.com/apikey

## Запуск

### Перевірка окремих модулів (без API)

```bash
python test_1.py   # BufferMemory — не потребує API
python test_2.py   # SummaryMemory — потребує API для тесту стискання
python test_3.py   # VectorMemory — потребує chromadb + sentence-transformers
```

### Повний експеримент

```bash
# Запуск всіх чотирьох тестів (~30-40 хвилин через rate limit)
python experiment.py

# Побудова графіків після завершення
python visualize.py
```

### Вихідні файли

| Файл | Опис |
|------|------|
| `results.json` | Всі числові результати експерименту |
| `results_partial.json` | Проміжні результати (оновлюється під час запуску) |
| `fig1_retention.png` | Retention Score vs кількість дистракторів |
| `fig2_latency.png` | Порівняння латентності (boxplot) |
| `fig3_context.png` | Зростання розміру контексту |
| `fig4_heatmap.png` | Retention по кожному питанню (теплова карта) |
| `fig5_stats.png` | Результати статистичного тесту |

## Стратегії пам'яті

| Стратегія | Принцип | Переваги | Недоліки |
|-----------|---------|----------|----------|
| BufferMemory | Ковзне вікно (останні N повідомлень) | Мінімальна латентність | Факти з початку розмови губляться |
| SummaryMemory | Стискання через Gemini | Зберігає ключові факти | Додаткові API-виклики |
| VectorMemory | Семантичний пошук через ChromaDB | Довгострокова пам'ять | Потребує локальної векторної БД |

## Технічний стек

- **LLM**: Google Gemini 1.5 Flash (безкоштовний API)
- **Embeddings**: `all-MiniLM-L6-v2` (384 виміри, sentence-transformers)
- **Векторна БД**: ChromaDB (in-memory, cosine similarity)
- **Статистика**: SciPy (Paired T-test / Wilcoxon)
- **Візуалізація**: Matplotlib, Seaborn

## Відтворення результатів

Для повного відтворення результатів:
1. Встановити залежності з `requirements.txt`
2. Додати `GEMINI_API_KEY` у `.env`
3. Запустити `python experiment.py`
4. Запустити `python visualize.py`

Результати можуть незначно відрізнятись через стохастичність Gemini API
(temperature=0.1 мінімізує варіативність, але не усуває повністю).
