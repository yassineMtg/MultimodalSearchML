#Multimodal Search ML  

---

#Milestone 1

---

##Business Idea as an ML Problem

In the fast-growing landscape of e-commerce, customers expect fast, accurate, and personalized search experiences. However, traditional keyword-based search engines often fall short when handling vague, visual, or ambiguous queries — such as "Nike shoes with blue swoosh" or "Apple logo hoodie" — where intent cannot be fully captured by text alone.

This project addresses this challenge by leveraging multimodal machine learning techniques to improve product search. Specifically, it uses both textual and visual data from the Shopping Queries Image Dataset (SQID) and extracts semantic embeddings via CLIP, a vision-language model.

The machine learning problem is framed as a multimodal ranking/retrieval task, where the system computes a semantic similarity score between a user query and multiple product candidates, each represented through both their text description and image. The goal is to return a ranked list of products most relevant to the user's intent.

This approach enables improved understanding of visual and textual attributes, making the search more intuitive, robust to vocabulary gaps, and capable of handling open-ended or visually grounded queries.

---

##Business Case

Modern e-commerce platforms face increasing pressure to deliver accurate and engaging search experiences. When users enter queries, they expect results that match both their functional and visual expectations. For example, a search for "slim fit striped shirt" should return products that actually have the right cut and visual pattern — not just any shirt with matching keywords.

However, most search engines still rely on traditional information retrieval methods that focus only on textual metadata. These methods often fail when:

    The product image contradicts the description.

    Visual details are not fully captured by the product title or description.

    The user’s intent is visual or ambiguous.

This project aims to improve the product search experience by using multimodal machine learning, combining product titles, descriptions, and images to better understand relevance. It leverages the Shopping Queries Image Dataset (SQID), an enriched dataset built on top of Amazon’s Shopping Queries Dataset (SQD), which includes product images and pretrained CLIP embeddings.

By understanding both the visual and textual context of each product, we can return more meaningful and satisfying search results — enhancing customer experience, increasing conversion rates, and reducing bounce rates.

---


