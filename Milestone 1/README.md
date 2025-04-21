# Multimodal Search ML  

---

# Milestone 1

## Business Idea as an ML Problem

In the fast-growing landscape of e-commerce, customers expect fast, accurate, and personalized search experiences. However, traditional keyword-based search engines often fall short when handling vague, visual, or ambiguous queries — such as "Nike shoes with blue swoosh" or "Apple logo hoodie" — where intent cannot be fully captured by text alone.

This project addresses this challenge by leveraging multimodal machine learning techniques to improve product search. Specifically, it uses both textual and visual data from the Shopping Queries Image Dataset (SQID) and extracts semantic embeddings via CLIP, a vision-language model.

The machine learning problem is framed as a multimodal ranking/retrieval task, where the system computes a semantic similarity score between a user query and multiple product candidates, each represented through both their text description and image. The goal is to return a ranked list of products most relevant to the user's intent.

This approach enables improved understanding of visual and textual attributes, making the search more intuitive, robust to vocabulary gaps, and capable of handling open-ended or visually grounded queries.

---

## Business Case

Modern e-commerce platforms face increasing pressure to deliver accurate and engaging search experiences. When users enter queries, they expect results that match both their functional and visual expectations. For example, a search for "slim fit striped shirt" should return products that actually have the right cut and visual pattern — not just any shirt with matching keywords.

However, most search engines still rely on traditional information retrieval methods that focus only on textual metadata. These methods often fail when:

- The product image contradicts the description.

- Visual details are not fully captured by the product title or description.

- The user’s intent is visual or ambiguous.

This project aims to improve the product search experience by using multimodal machine learning, combining product titles, descriptions, and images to better understand relevance. It leverages the Shopping Queries Image Dataset (SQID), an enriched dataset built on top of Amazon’s Shopping Queries Dataset (SQD), which includes product images and pretrained CLIP embeddings.

By understanding both the visual and textual context of each product, we can return more meaningful and satisfying search results — enhancing customer experience, increasing conversion rates, and reducing bounce rates.

---

## Business Value 

Using machine learning, particularly multimodal models, provides substantial business value for modern e-commerce platforms. Traditional search engines based on keyword matching often fail to capture subtle user intent or visual expectations, leading to irrelevant results and poor user experience.

By introducing CLIP-based multimodal search, the system gains the ability to:

- Interpret complex queries more semantically, not just syntactically.

- Leverage both product images and text to rank relevance more effectively.

- Adapt to user preferences over time without manual rule-based tuning.

In summary, applying machine learning in this context helps companies stay competitive by improving user satisfaction, driving revenue, and enabling smarter automation in product search and recommendation systems.

---

## Data Overview

This project uses the Shopping Queries Image Dataset (SQID), an extension of the original Shopping Queries Dataset (SQD) released by Amazon for the KDDCup’22 challenge. SQID enriches the SQD dataset by including product images and precomputed CLIP embeddings, allowing for multimodal learning and ranking tasks.

-----

Key dataset components:

- Queries: User search terms in natural language (e.g., "men’s red sneakers").

- Products: Product listings with titles, descriptions, brand, color, etc.

- Images: Main product image for each item, linked via URL.

- Labels (ESCI): Relevance judgments:

  - E – Exact match

  - S – Substitute

  - C – Complement

  - I – Irrelevant

- Embeddings: Precomputed CLIP text and image embeddings for each product.

Dataset stats:

- ~190,000 product listings

- ~1.1M query-product pairs

- Covers US, ES, and JP locales (this project focuses on the US locale)

The dataset is publicly available at:
🔗 [https://github.com/Crossing-Minds/shopping-queries-image-dataset](https://github.com/Crossing-Minds/shopping-queries-image-dataset)

---

## Project Archetype

Based on the archetype framework taught in class, this project falls under the Software 2.0 category.

Software 2.0 refers to systems where traditional code is replaced or augmented by machine-learned logic — typically models trained on data. Instead of writing explicit rules for ranking products or understanding user queries, the system learns from examples and uses embeddings (text + image) to drive behavior.

In this project:

- The ranking function is learned through semantic similarity between query and product embeddings.

- The rules for what is relevant are not hardcoded — they are inferred from CLIP’s understanding of visual-textual alignment.

- The system adapts and improves based on the distribution of queries and product listings.

This is in contrast to:

- Autonomous systems: which act in the world (e.g., robotics).

- Human-in-the-loop: where a person is part of the feedback cycle (e.g., labeling data or approving results).

Thus, this project is a clear example of Software 2.0 — where ML replaces handcrafted relevance ranking logic.

---

## Feasibility Analysis

### Literature Review

This project is inspired by two key papers:

- Shopping Queries Dataset (SQD) – [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)

  - Introduced by Amazon, this dataset provides a large-scale benchmark for product search using query-product pairs labeled by relevance. It defined the ESCI labels (Exact, Substitute, Complement, Irrelevant) and proposed ranking, classification, and substitution detection tasks.

- Shopping Queries Image Dataset (SQID) – [arXiv:2405.15190](https://arxiv.org/abs/2405.15190)

  - SQID builds on SQD by enriching it with product images and pretrained embeddings (text and image) using CLIP. It supports multimodal learning and highlights the value of combining text + image for better product ranking.

### Baseline Model

**Baseline Model Summary**

| Model Name                             | Developer                                    | Purpose                                      | Performance (NDCG) |
|----------------------------------------|----------------------------------------------|----------------------------------------------|--------------------|
| CLIP (ViT-L/14)                        | OpenAI                                       | Multimodal embedding for image-text matching | ~0.82 (SQID paper) |
| SBERT (MiniLM-L12-v2)                  | Hugging Face                                 | Text-only semantic similarity model          | ~0.83              |
| ESCI Baseline (MS MARCO Cross-Encoder) | Amazon                                       | Fine-tuned text-only ranker for SQD          | **0.85+**          |

The baseline model used in this project is CLIP (Contrastive Language-Image Pretraining), specifically the clip-vit-large-patch14 variant, available via Hugging Face.

- It provides joint embeddings for both text (user queries) and images (product photos).

- Embeddings can be compared using cosine similarity to estimate semantic relevance.

This model is:

- Open-source

- Available as a pretrained binary

- Does not require retraining to get useful results (zero-shot setup)

---

## Evaluation Metrics

This project focuses on ranking product search results by relevance to a given user query. Therefore, the most appropriate evaluation metrics are:

### 1. Normalized Discounted Cumulative Gain (NDCG)

- Why?: NDCG measures the quality of a ranked list by rewarding correct items at higher positions.

- How it works:

  - Takes into account the position of relevant items in the list.

  - ESCI labels are mapped to graded relevance scores:

    - E (Exact) = 1.0

    - S (Substitute) = 0.1

    - C (Complement) = 0.01

    - I (Irrelevant) = 0.0

  - Used in: The official SQD and SQID benchmarks.

### 2. Precision @ K (Optional)

- Measures how many of the top-K retrieved products are truly relevant (e.g., Exact or Substitute).

- Useful for evaluating short result lists, like top 5 or top 10.

### 3. Cosine Similarity (Intermediate)

- Not a final metric, but used internally to compare query and product embeddings (via CLIP).

- It’s how we score and rank product candidates before computing NDCG.

These metrics provide a robust framework to evaluate how well the system understands and ranks product relevance based on multimodal inputs.

---


