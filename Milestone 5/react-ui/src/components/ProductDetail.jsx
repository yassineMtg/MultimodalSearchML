// src/components/ProductDetail.jsx
import React, { useEffect, useState } from 'react';
import './ProductDetail.scss';

const ProductDetail = () => {
  const [product, setProduct] = useState(null);

  useEffect(() => {
    try {
      const searchParams = new URLSearchParams(window.location.search);
      const rawData = searchParams.get("data");
      if (!rawData) throw new Error("No data param");

      const parsed = JSON.parse(rawData);
      setProduct(parsed);
    } catch (err) {
      console.error("Error parsing product data", err);
    }
  }, []);

  const stripHtml = (html) => {
    const div = document.createElement("div");
    div.innerHTML = html.replaceAll("<br/>", "\n");
    return div.textContent || div.innerText || "";
  };

  if (!product) {
    return <div className="product-detail"><p>Invalid or missing product data.</p></div>;
  }

  return (
    <div className="product-detail">
      <div className="product-card">
        {product.image_urls && (
          <img
            src={product.image_urls.split(",")[0]?.trim()}
            alt={product.product_title}
            className="product-image"
          />
        )}

        <h2 className="title">{product.product_title}</h2>

        <p><strong>Brand:</strong> {product.product_brand || "N/A"}</p>
        <p><strong>Color:</strong> {product.product_color || "N/A"}</p>
        <p><strong>Score:</strong> {(product.score * 100).toFixed(2)}%</p>

        {product.product_description && (
          <div className="description-block">
            <strong>Description:</strong>
            <pre className="description">
              {stripHtml(product.product_description)}
            </pre>
          </div>
        )}

        <div className="external-link">
          <a
            href={`https://www.amazon.com/dp/${product.product_id}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            🔗 View this product on Amazon
          </a>
        </div>
      </div>
    </div>
  );
};

export default ProductDetail;
