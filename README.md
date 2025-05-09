# adkllama - RAG Agent

Это репозиторий для проекта RAG (Retrieval-Augmented Generation) агента, использующего LlamaIndex, Qdrant в качестве векторной базы данных и модели OpenAI (или совместимые) для генерации ответов.

## О проекте

Проект демонстрирует создание и использование гибридного ретривера, который комбинирует векторный поиск (с использованием Qdrant) и BM25 поиск для улучшения релевантности retrieved документов. Полученные документы затем используются языковой моделью для генерации ответа на запрос.

Ключевые компоненты:
- **Qdrant**: Высокопроизводительная векторная база данных.
- **LlamaIndex**: Фреймворк для построения приложений на LLM с возможностью интеграции внешних данных.
- **Hybrid Retrieval**: Комбинация векторного и полнотекстового поиска для более точного извлечения информации.
- **Reranking**: Переупорядочивание результатов поиска для улучшения их релевантности.

## Начало работы

Для запуска проекта необходимо настроить окружение и установить зависимости.

### Предпосылки

- Python 3.8+
- Учетная запись в Qdrant Cloud или локальный экземпляр Qdrant.
- Ключ API для OpenAI или другой совместимой LLM.

### Установка

1.  Клонируйте репозиторий:
    ```bash
    git clone <URL вашего репозитория>
    cd adkllama
    ```
2.  Создайте виртуальное окружение (рекомендуется):
    ```bash
    python -m venv venv
    source venv/bin/activate # Для Linux/macOS
    # venv\Scripts\activate # Для Windows
    ```
3.  Установите зависимости. **Важно:** Убедитесь, что ваша установка `pip` и `setuptools` совместима с версией Python (3.13.3). Если возникнут ошибки при установке, возможно, потребуется обновить или пересоздать виртуальное окружение.
    ```bash
    pip install -r requirements.txt
    ```

### Конфигурация

Проект использует переменные окружения для хранения конфиденциальных данных, таких как ключи API.

1.  Создайте файл `.env` в корневой директории проекта.
2.  Добавьте в него следующие строки, заменив значения своими учетными данными:
    ```dotenv
    OPENAI_API_KEY="ВАШ_КЛЮЧ_OPENAI"
    QDRANT_URL="URL_ВАШЕГО_QDRANT_INSTANCE"
    QDRANT_API_KEY="ВАШ_КЛЮЧ_API_QDRANT"
    ```
    **Важно:** Файл `.env` добавлен в `.gitignore` и не будет отслеживаться Git. **Никогда не добавляйте ваши секретные ключи напрямую в код или в файлы, отслеживаемые Git.**

### Запуск

Основной исполняемый файл проекта - Jupyter Notebook `rag_agent/Untitled.ipynb`.

1.  Убедитесь, что вы находитесь в активированном виртуальном окружении.
2.  Запустите Jupyter Lab или Jupyter Notebook в корневой директории проекта:
    ```bash
    jupyter lab
    # или
    jupyter notebook
    ```
3.  Откройте файл `rag_agent/Untitled.ipynb` и последовательно выполните ячейки.

## Структура проекта

```
├── rag_agent/
│   ├── __init__.py
│   ├── agent.py             # Core RAG agent logic
│   └── vector_search/
│       ├── __init__.py
│       ├── custom_query_engine_tool.py
│       ├── document_loader.py
│       ├── hybrid_retriever.py
│       ├── qdrant_vector_store.py
│       ├── reranker.py
│       └── utils.py
├── examples/               # Example usage of the RAG agent
│   └── main.ipynb          # Example usage for the agent
├── tests/                  # Test suite for the project
│   ├── __init__.py         # Mark tests as a package
│   └── test_agent.py #test for the agent.py
│   └── rag_agent/
│       └── vector_search/
│           ├── test_custom_query_engine_tool.py
│           ├── test_document_loader.py
│           ├── test_hybrid_retriever.py
│           ├── test_qdrant_vector_store.py
│           ├── test_reranker.py
│           └── test_utils.py
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── .gitignore                # Version control ignore list
└── .env.example              # Example environment file

```
