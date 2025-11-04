# Chroma Quotas

Chroma Cloud comes with quotas. Here is a list of the default values they provide and what they mean.

## ToC

1. [Add / Update / Upsert](#add--update--upsert)
2. [Query / Get / Delete](#query--get--delete)
3. [Query](#query)
4. [Collections Create / Update](#collections-create--update)

---

# Add / Update / Upsert

## Max Embedding Dimensionality

- Default value: `4096`

This is the maximum size of the embedding vector you can store. Most embedding models use dimensions of 384, 768, 1536, or 3072. The value of 4096 is very safe and supports even the most advanced models (e.g., new versions of OpenAI embeddings can have up to 3072 dimensions). For a Memory service, this default value is appropriate.

In other words this is the amount of dimensions each document in Chroma has. Typical two-dimensional value is `[0.23, -0.15]`.

384-dimensional vector has 384 values: `[0.001, 0.002, 0.003, ..., 0.383, 0.384]`

## Max Document Size in Bytes

- Default value: `16384`

This is the maximum size of a single text document you can store. 16KB is approximately 4000-5000 words in English language. One English character from the UTF-8 code page costs **1 byte**. For example, word `Hello` costs **5 bytes**.

Characters with diacritics typically cost **2 bytes** (Latin scripts) or **3 bytes** (complex diacritics). For example, Vietnamese "Hello world" is `Chào thế giới` which costs:

<table>
  <thead>
    <tr>
      <th>Character</th>
      <th>Bytes</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>C</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>h</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>à</code></td>
      <td>2</td>
      <td>Diacritical latin</td>
    </tr>
    <tr>
      <td><code>o</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td>(space)</td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>t</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>h</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>ế</code></td>
      <td><strong>3</strong></td>
      <td>Two diacriticals: circumflex + acute</td>
    </tr>
    <tr>
      <td>(space)</td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>g</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>i</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><code>ớ</code></td>
      <td><strong>3</strong></td>
      <td>o + horn + acute</td>
    </tr>
    <tr>
      <td><code>i</code></td>
      <td>1</td>
      <td>ASCII</td>
    </tr>
    <tr>
      <td><strong>TOTAL</strong></td>
      <td><strong>18</strong></td>
      <td></td>
    </tr>
  </tbody>
</table>

> **NOTE:** Every single Emoji in text costs **4 bytes**!
>
> String "Hello 👋" costs: 5 + 1 + 4 = **10 bytes!**

Another example is [Agile Manifesto](https://agilemanifesto.org/). This page contains a lot of translations. So let's see which language is the "cheapest":

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Language</th>
      <th>Bytes</th>
      <th>Characters</th>
      <th>Bytes/Char</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🥇</td>
      <td><strong>Chinese Simplified 🇨🇳</strong></td>
      <td><strong>361</strong></td>
      <td>131</td>
      <td>2.76</td>
      <td>✨ <strong>CHEAPEST</strong></td>
    </tr>
    <tr>
      <td>🥈</td>
      <td>Chinese Traditional 🇹🇼</td>
      <td>393</td>
      <td>143</td>
      <td>2.75</td>
      <td></td>
    </tr>
    <tr>
      <td>🥉</td>
      <td>English 🇬🇧</td>
      <td>422</td>
      <td>422</td>
      <td>1.00</td>
      <td>📌 Baseline (ASCII)</td>
    </tr>
    <tr>
      <td>4️⃣</td>
      <td>Spanish 🇪🇸</td>
      <td>482</td>
      <td>477</td>
      <td>1.01</td>
      <td></td>
    </tr>
    <tr>
      <td>5️⃣</td>
      <td>Arabic 🇸🇦</td>
      <td>487</td>
      <td>266</td>
      <td>1.83</td>
      <td></td>
    </tr>
    <tr>
      <td>6️⃣</td>
      <td><strong>Japanese 🇯🇵</strong></td>
      <td><strong>634</strong></td>
      <td>218</td>
      <td>2.91</td>
      <td>💸 <strong>MOST EXPENSIVE</strong></td>
    </tr>
  </tbody>
</table>

```
Chinese Simplified 🇨🇳  ████████████████████████████ 361 bytes ✨ WINNER
Chinese Traditional 🇹🇼 ██████████████████████████████ 393 bytes
English 🇬🇧             █████████████████████████████████ 422 bytes
Spanish 🇪🇸             ██████████████████████████████████████ 482 bytes
Arabic 🇸🇦              ██████████████████████████████████████ 487 bytes
Japanese 🇯🇵            ██████████████████████████████████████████████████ 634 bytes 💸 MOST EXPENSIVE
```

**Key Insight**: Chinese Simplified is the most byte-efficient (361 bytes) while Japanese is the least efficient (634 bytes) - a 75.6% difference!

**Why?** Chinese uses the fewest characters (131) despite 2.76 bytes/char, while Japanese uses 218 characters at 2.91 bytes/char.

Why Chinese Simplified won? `软件`(2) < `software`(8). These two words mean the same. Two Chinese characters costing nearly 3 bytes each, but in total it's 6 bytes instead of English 8 bytes.

Why English didn't win? It's the best and most effective encoding. Each character belongs to ASCII table and costs only 1 byte. However, Agile Manifesto has 422 characters in total. Some long words like "comprehensive" and "documentation" don't help.

Spanish is one of the Latin alphabets, making it closely related to English. However, some diacritical letters like `ó, á, í, ñ` cost 2 bytes each and some Spanish words in Agile Manifesto are even longer than in English.

Arabic has only 266 characters, but unfortunately each Arabic character costs 2 bytes and some diacritical ones even more, making it 487 bytes heavy.

Japanese is the biggest loser here:

- Kanji (Chinese chars) cost 3 bytes each
- Hiragana (Japanese syllables) cost 3 bytes each too
- Katakana (foreign words) - they are costly too! 3 bytes each

As if it wasn't enough - Japanese sometimes uses more characters than Chinese. For example, "Software Development" is:

- `软件开发` in Chinese (4 characters)
- `ソフトウェア開発` in Japanese (8 characters)

Key takeaways:

- Asian languages (Chinese, Japanese) use ~3 bytes per character
- European languages use on average 1.0-1.8 bytes per character
- English uses exactly 1 byte per character

If you want to store one English memory worth of 16 kB, you might need 32 kB for mixed languages and even 64 kB for Asian languages!

> Think what language you're using when you store your memories!

## Max URI Size in Bytes

- Default value: `256`

Maximum size of URI/URL that you can store as metadata. 256 bytes is a standard URL length. For a Memory system, you probably won't store many URIs, so keep it or decrease to 128 if you want to save resources.

Usually you won't be storing URL as metadata like this:

```json
{
  "document": "User mentioned loving pizza",
  "metadata": {
    "user": "John",
    "source_url": "https://slack.com/messages/C123456/p1234567890"
  }
}
```

For memories in Claude you'll rarely use this feature so keep it at `256` as it's more than enough.

But anyway - if you wonder how much is 254-characters-long URL? Here's an example:

`https://my-bucket.s3.eu-west-1.amazonaws.com/projects/network-infrastructure/chromadb-config.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE12345&X-Amz-Date=20241104T120000Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=abc123`

So you can decide on your own.

## Max ID Size in Bytes

- Default value: `128`

Maximum size of the document identifier. 128 bytes is plenty for an ID (you can fit UUID + timestamp + prefix).

Examples:

```
"550e8400-e29b-41d4-a716-446655440000"  = 36 characters = 36 bytes
"memory_550e8400-e29b-41d4-a716-446655440000"  = 43 characters = 43 bytes
"user_john_2024-11-04_550e8400-e29b-41d4-a716-446655440000"  = 57 characters = 57 bytes
"memory_user_john_project_enigma_date_20241104_uuid_550e8400-e29b-41d4-a716-446655440000"  = 87 characters = 87 bytes
"memory_user_john_smith_project_chromadb_category_technical_date_20241104_120530_uuid_550e8400-e29b-41d4-a716-446655440000" = 121 characters = 121 bytes
```

## Max Metadata Size in Bytes

- Default value: `4096`

Total size of all metadata for a single document during add/update/upsert operations. 4KB is quite a lot of space for metadata (user_id, timestamp, conversation_id, tags, etc.). For Memory, this should be sufficient, but if you plan to store rich metadata (e.g., long tags, context), consider increasing to 8192.

### Example 1 (200-250 bytes)

Document:

```
"User John loves pizza and prefers thin crust"
```

Its metadata:

```json
{
  "user_id": "john_smith",
  "timestamp": "2024-11-04T12:30:00Z",
  "conversation_id": "conv_12345",
  "category": "food_preferences",
  "importance": "high",
  "tags": ["food", "pizza", "preferences"],
  "source": "chat",
  "language": "en"
}
```

### Example 2 (~1000 bytes)

```json
{
  "user_id": "john_smith_employee_12345",
  "user_email": "john.smith@company.com",
  "timestamp": "2024-11-04T12:30:45.123Z",
  "conversation_id": "conv_550e8400-e29b-41d4-a716-446655440000",
  "project_id": "project_netbox_etl_automation",
  "category": "technical_discussion",
  "subcategory": "server_infrastructure",
  "importance": "high",
  "confidence_score": 0.95,
  "tags": [
    "netbox",
    "ansible",
    "automation",
    "switzerland",
    "device_type_library",
    "etl",
    "network_equipment",
    "cisco",
    "data_enrichment"
  ],
  "related_documents": ["doc_123", "doc_456", "doc_789"],
  "source": "claude_chat",
  "platform": "claude.ai",
  "language": "en",
  "detected_entities": ["NetBox", "Ansible", "Switzerland", "Cisco"],
  "sentiment": "neutral",
  "context": "User discussing NetBox ETL workflow for Swiss network equipment",
  "previous_conversation_refs": ["conv_abc123", "conv_def456"],
  "team": "EMEA_GLOBAL_NET"
}
```

### Example 3 (~3500 bytes)

```json
{
  "user_id": "john_smith_employee_12345",
  "timestamp": "2024-11-04T12:30:45.123456Z",
  "conversation_id": "conv_550e8400-e29b-41d4-a716-446655440000",
  "tags": ["tag1", "tag2", "tag3", ... "tag50"],  // 50 tags
  "context_summary": "User was discussing implementation of a microservices architecture for an e-commerce platform. The conversation covered API gateway configuration, database sharding strategies, Redis caching layers, and Docker containerization. User mentioned challenges with payment processing integration, inventory management synchronization, and handling peak traffic during seasonal sales. Discussion also touched on security best practices, GDPR compliance requirements, and monitoring solutions using Prometheus and Grafana...",  // ~500 chars
  "related_documents": ["doc_001", "doc_002", ... "doc_100"],  // 100 documents
  "detected_entities": ["entity1", "entity2", ... "entity30"],
  "full_context": "Very long context string with detailed information about the entire conversation flow, including all mentioned topics, decisions made, action items discussed, and references to previous conversations..."  // another ~1000 chars
}
```

## Max Metadata Key Size in Bytes

- Default value: `36`

Maximum size of a single metadata key name. 36 bytes is enough for descriptive keys like "conversation_timestamp" or "user_identification_tag".

Example key size length:

```json
{
  "user": "john", // 4 chars
  "timestamp": "2024-11-04", // 9 chars
  "project": "netbox", // 7 chars
  "category": "technical", // 8 chars
  "tags": ["python", "automation"] // 4 chars
}
```

Is 36 enough for you? Have a look at this example and decide for yourself:

```json
{
  "user_conversation_metadata_source": "chat", // 33 chars
  "project_related_conversation_topic": "automation", // 35 chars
  "conversation_timestamp_utc_format": "2024-11-04" // 33 chars
}
```

## Max Number of Metadata Keys

- Default value: `32`

Maximum number of different metadata keys per document. 32 keys is a lot. A typical Memory system might use 5-10 keys (user_id, timestamp, project_id, category, importance, etc.).

### Example with 10 keys

```json
{
  "user": "john",
  "timestamp": "2024-11-04T12:30:00Z",
  "project": "netbox_etl",
  "category": "technical",
  "subcategory": "automation",
  "importance": "high",
  "confidence": 0.95,
  "tags": ["python", "ansible"],
  "source": "chat",
  "language": "en"
}
```

### Extreme 32 keys example that hits the limit

```json
{
  "user": "john",
  "user_email": "john@company.com",
  "user_role": "engineer",
  "timestamp": "2024-11-04T12:30:00Z",
  "created_at": "2024-11-04",
  "updated_at": "2024-11-04",
  "conversation_id": "conv_12345",
  "project": "netbox_etl",
  "project_phase": "development",
  "project_status": "active",
  "category": "technical",
  "subcategory": "automation",
  "topic": "infrastructure",
  "importance": "high",
  "priority": 1,
  "urgency": "medium",
  "confidence": 0.95,
  "sentiment": "neutral",
  "emotion": "focused",
  "tags": ["python", "ansible"],
  "keywords": ["automation", "etl"],
  "source": "chat",
  "platform": "claude.ai",
  "language": "en",
  "detected_entities": ["NetBox"],
  "related_docs": ["doc_123"],
  "team": "EMEA_GLOBAL_NET",
  "department": "IT",
  "location": "Switzerland",
  "version": "1.0",
  "schema_version": "2024-11",
  "is_archived": false
}
```

---

# Query / Get / Delete

## Max ID Size in Bytes

- Default value: `128`

Same as the one in [Add/Update/Upsert](#add-update-upsert). Keep it consistent with the write value.

Perhaps Chroma Devs should remove this value as creating id `550e8400-e29b-41d4-a716-446655440000` with the length of `36` bytes and then having a Query/Get/Delete limit of let's say `32` bytes will result as error: `Error: ID exceeds maximum length of 32 bytes`. You won't be able to Search, Get and Delete this document (memory).

## Max Number of Where Predicates

- Default value: `8`

Maximum number of filtering conditions in a WHERE query. For example: `where={"user": "John", "project": "X", "date": {...}}` is 3 predicates. 8 predicates is usually sufficient for complex queries like "find all memories for user John from project X, created in the last week, with category 'technical', with tag 'urgent'". Keep it or increase to 12-16 if you plan very complex filters.

Long story short - having this document in Chroma:

```json
{
  "document": "User John loves pizza with thin crust",
  "metadata": {
    "user": "john",
    "category": "food",
    "importance": "high",
    "timestamp": "2024-11-04"
  }
}
```

It contains 4 meta keys. Having `Max Number of Where Predicates` == `2` means you can search only using **two arguments**:

```python
results = collection.query(
    query_texts=["pizza preferences"],
    where={
        "user": "john",      # Argument 1
        "category": "food"   # Argument 2
    },
    n_results=5
)
```

Or:

```python
results = collection.query(
    query_texts=["pizza preferences"],
    where={
        "importance": "high",  # Argument 1
        "category": "food"     # Argument 2
    },
    n_results=5
)
```

## Max Where Size in Bytes

- Default value: `4096`

Total size of the WHERE expression in bytes. 4KB allows for very elaborate conditions.

Example of ~150 bytes WHERE query:

```python
where={
    "user": "john",
    "project": "netbox_etl",
    "category": "technical"
}
```

Example of ~500 bytes WHERE:

```python
where={
    "user": "john_smith_employee_12345",
    "project": "netbox_etl_automation_system",
    "category": "technical_infrastructure",
    "subcategory": "network_automation",
    "importance": "high_priority",
    "created_at": {"$gte": "2024-11-01T00:00:00Z", "$lte": "2024-11-04T23:59:59Z"},
    "tags": {"$in": ["python", "ansible", "automation", "netbox"]},
    "team": "EMEA_GI_Network_Infrastructure"
}
```

## Max Number of Where Document Predicates

- Default value: `8`

Number of predicates you can use to filter by document content (not metadata, but the text itself). This is useful if you want to search for memories containing specific phrases.

### Difference between "metadata's WHERE" and "document's WHERE"

Metadata's WHERE:

```python
where={
    "user": "john",      # This is in metadata
    "project": "netbox"  # This as well
}
```

WHERE **Document**:

```python
where_document={
    "$contains": "pizza"  # Searches in the document's content
}
```

### Example of a simple search

The document:

```json
{
  "document": "User John loves pizza with thin crust and extra cheese",
  "metadata": {
    "user": "john",
    "category": "food"
  }
}
```

Meta WHERE search:

```python
collection.query(
    query_texts=["food"],
    where={"user": "john"}  # Meta search
)
```

Document WHERE search:

```python
collection.query(
    query_texts=["food"],
    where_document={"$contains": "pizza"}  # Searching in the document content
)
```

### Example of the default max 8 doc's predicates

```python
collection.query(
    query_texts=["memories"],
    where_document={
        "$and": [
            {"$contains": "pizza"},          # Predicate 1
            {"$contains": "cheese"},         # Predicate 2
            {"$contains": "thin crust"},     # Predicate 3
            {"$contains": "italian"},        # Predicate 4
            {"$contains": "restaurant"},     # Predicate 5
            {"$not_contains": "pineapple"},  # Predicate 6
            {"$not_contains": "anchovies"},  # Predicate 7
            {"$contains": "margherita"}      # Predicate 8
        ]
    }
)
```

## Max Where Document Size in Bytes

- Default value: `130`

Max length of document's WHERE search phrase. For example:

```python
where_document = {
    "$contains": "NetBox automation system"  # 25 chars = 25 bytes
}
```

---

# Query

## Max number of query results returned

- Default value: `300`

Maximum number of results that a single query can return. For a Memory system, this is a good setting. Usually you'll need 5-20 most relevant memories, but 300 gives flexibility for pagination and bulk operations.

Query for 5 results:

```python
results = collection.query(
    query_texts=["Project management"],
    n_results=5  # I want top 5 results
)
```

Query for 300 results:

```python
results = collection.query(
    query_texts=["Project management"],
    n_results=300  # I want top 300 results
)
```

---

# Collections Create / Update

## Max Metadata Size in Bytes

- Default value: `256`

Size of metadata for the collection itself (not documents). This is used to describe the entire collection, e.g., {"description": "User memories", "version": "1.0"}.

It's the length of total amount of collection's metadata. The collection's description you'll see when you log into Chroma Cloud.

This description:

```python
metadata = {
    "description": "MCP Memory System - User memories, preferences, and conversation history",
    "version": "1.0.0",
    "created_at": "2024-11-04T12:00:00Z",
    "created_by": "john_smith",
    "purpose": "AI agent memory storage",
    "schema_version": "2024-11"
}
```

Will result in as JSON's ~85 bytes as:

```json
{
  "description": "MCP Memory System - User memories, preferences, and conversation history",
  "version": "1.0.0",
  "created_at": "2024-11-04T12:00:00Z",
  "created_by": "john_smith",
  "purpose": "AI agent memory storage",
  "schema_version": "2024-11"
}
```

## Max Metadata Key Size in Bytes

- Default value: `36`

Size of collection metadata key names. Similar to Document's "Max Metadata Key Size in Bytes" but this time related to collection's meta.

```python
metadata = {
    "collection_description_long": "...",        # 28 chars
    "collection_metadata_schema_ver": "...",     # 31 chars
    "collection_created_timestamp_utc": "..."    # 33 chars
}
```

## Max Number of Metadata Keys

- Default value: `16`

Number of metadata keys for the collection. Similar to Document's "Max Number of Metadata Keys but this time related to collection's meta.

### Three meta keys example

```python
metadata = {
    "description": "User memories and preferences",  # 1
    "version": "1.0",                                # 2
    "created_at": "2024-11-04"                       # 3
}
```
