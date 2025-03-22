# Multimodal Search ML  

---

# Milestone 1

---

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

-Autonomous systems: which act in the world (e.g., robotics).

-Human-in-the-loop: where a person is part of the feedback cycle (e.g., labeling data or approving results).

Thus, this project is a clear example of Software 2.0 — where ML replaces handcrafted relevance ranking logic.





