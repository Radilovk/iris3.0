# KV Keys за binding `iris_rag_kv`

Тази папка съдържа всички KV ключове, които трябва да се създадат в Cloudflare KV namespace `iris_rag_kv`
(ID: `1fd2ab63b3634a1f8f0627c6fde76fd9`).

## Как да създадете ключовете

Използвайте Wrangler CLI:

```bash
wrangler kv key put --namespace-id=1fd2ab63b3634a1f8f0627c6fde76fd9 "config:ai" "$(cat kv-keys/config__ai.json)"
```

Или чрез Admin панела на приложението (`/admin/config` — POST заявка).

## Списък на ключовете

| Файл | KV ключ | Описание |
|------|---------|----------|
| `config__ai.json` | `config:ai` | AI конфигурация — провайдер, модел, API URL-и |

> **Забележка:** Ключовете `result:<side>:<hash>:<model>` се създават автоматично от pipeline-а при анализ на изображение и имат TTL от 24 часа. Те не изискват ръчно създаване.
