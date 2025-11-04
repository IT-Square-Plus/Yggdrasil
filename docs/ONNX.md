# ONNX Model with ChromaDB

## What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open standard for representing machine learning models. It enables transferring models between different AI frameworks (PyTorch, TensorFlow, scikit-learn) and running them in optimized runtime.

## Why does ChromaDB use ONNX?

ChromaDB needs ONNX for **semantic search** - searching based on meaning, not just keywords.

### How it works:

1. **Text → Embedding**

   - Your memory: "Python 3.13 released with improved type hints"
   - ONNX model (`all-MiniLM-L6-v2`) converts this to a vector of 384 numbers
   - This vector represents the **meaning** of the text

2. **Semantic search**
   - Query: "What's new in Python?"
   - ONNX converts query to a vector
   - ChromaDB compares vectors (cosine similarity)
   - Finds semantically similar memories, even if they use different words!

### Example:

```
Stored: "Python 3.13 released with improved type hints"
Query: "Latest Python version features"
```

**Traditional search:** No results (different words)
**Semantic search (ONNX):** ✅ Found! (similar meaning)

## Model: all-MiniLM-L6-v2

ChromaDB uses **all-MiniLM-L6-v2** by default:

- **Size:** ~80MB
- **Embedding dimension:** 384
- **Language:** Primarily English (works with Polish too, but less accurate)
- **Speed:** Very fast (~14ms/sentence on CPU)
- **Quality:** Good balance between size and accuracy

### Technical details:

- **Base model:** Microsoft MiniLM
- **Training:** Sentence Transformers
- **Format:** ONNX Runtime for optimization
- **Cache location:** `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`

## What happens on first startup?

1. **ChromaDB checks cache**

   - Location: `~/.cache/chroma/onnx_models/`

2. **If model not found:**

   - Downloads `all-MiniLM-L6-v2` from Hugging Face (~80MB)
   - Extracts to cache
   - Time: ~20 seconds (depends on internet)

3. **Subsequent startups:**
   - Model is in cache
   - Instant start ✅

### Logs on first startup:

```
⏱️  Chroma DB warming up! Waiting for ONNX embedding model...
[Progress bar 0% → 100%]
🤖 ONNX embedding model successfully downloaded!
```

## More information

- **ChromaDB Embeddings:** https://docs.trychroma.com/guides/embeddings
- **ONNX Runtime:** https://onnxruntime.ai/
- **Sentence Transformers:** https://www.sbert.net/
- **Model card:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

**Summary:** ONNX is the AI engine that converts text into numerical vectors, enabling ChromaDB to perform intelligent semantic search instead of simple keyword matching. This is core technology for the memory system!
