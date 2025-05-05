// src/components/SearchBox.jsx
import React, { useState } from 'react';
import axios from "axios";
import './SearchBox.scss';
import { FaTimes } from 'react-icons/fa';

const ITEMS_PER_PAGE = 30;

const stripHtml = (html) => {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent || div.innerText || "";
};

const truncateWords = (text, limit) => {
  if (!text) return '';
  const words = text.split(" ");
  return words.length > limit ? words.slice(0, limit).join(" ") + "..." : text;
};

const SearchBox = () => {
  const [query, setQuery] = useState("");
  const [image, setImage] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [topK, setTopK] = useState(100);
  const [threshold, setThreshold] = useState(0.8);
  const [allProducts, setAllProducts] = useState(true); // checked by default
  const [ignoreScore, setIgnoreScore] = useState(true); // checked by default

  const fetchResults = async () => {
    if (!query.trim() && !image) return;
    setLoading(true);

    try {
      let response;

      if (image) {
        const formData = new FormData();
        formData.append("image", image);
        formData.append("k", allProducts ? 9999 : topK);

        response = await axios.post("https://yassinemtg-smartsearch-api.hf.space/predict-image", formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 50000
        });
      } else {
        response = await axios.post("https://yassinemtg-smartsearch-api.hf.space/predict", {
          query,
          k: allProducts ? 9999 : topK
        }, { timeout: 50000 });
      }

      const filtered = ignoreScore
        ? response.data.results
        : response.data.results.filter(r => r.score >= threshold);

      setResults(filtered.slice(0, allProducts ? filtered.length : topK));
      setPage(1);
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    }

    setLoading(false);
  };

  const clearImage = () => {
    setImage(null);
  };

  const paginatedResults = results.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE);
  const totalPages = Math.ceil(results.length / ITEMS_PER_PAGE);

  const getScoreClass = (score) => {
    if (score >= 0.8) return "green";
    if (score >= 0.6) return "blue";
    if (score >= 0.5) return "orange";
    return "red";
  };

  return (
    <div className="container">
      <div className="inner">
        <h1>🔍 Smart Product Search</h1>

        <div className="controls-row">
          {/* <div className="search-upload-wrapper">
            <input
              type="text"
              className="search-input"
              placeholder="Search for a product (e.g. wireless headphones)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchResults()}
              title="Write what you want to search"
            />
            <label className="upload-btn">
              {image ? (
                <span className="file-ext">{'.' + image.name.split('.').pop()}</span>
              ) : (
                "📷"
              )}
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setImage(e.target.files[0])}
                hidden
              />
            </label>

          </div> */}

          <div className="search-upload-wrapper">
            <input
              type="text"
              className="search-input"
              placeholder="Search for a product (e.g. wireless headphones)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchResults()}
              title="Write what you want to search"
            />
            <div className="upload-container">
              <label className="upload-btn">
                {image ? (
                  <span className="file-ext">{'.' + image.name.split('.').pop()}</span>
                ) : (
                  "📷"
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setImage(e.target.files[0])}
                  hidden
                />
              </label>
              {image && (
                <button className="clear-image-btn" onClick={clearImage} title="Remove image">
                  <FaTimes />
                </button>
              )}
            </div>
          </div>
          

          <div className="input-group">
            <input
              type="number"
              className="tiny-input"
              min={1}
              value={topK}
              readOnly={allProducts}
              style={{
                cursor: allProducts ? "not-allowed" : "text",
                backgroundColor: allProducts ? "#ccc" : "#1f1f25",
                color: "white"
              }}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              title="Maximum number of products to show"
            />
            <label title="Show all products without limiting count">
              <input
                type="checkbox"
                checked={allProducts}
                onChange={() => setAllProducts(!allProducts)}
              />
              All
            </label>
          </div>

          <div className="input-group">
            <input
              type="number"
              step={0.01}
              min={0}
              max={1}
              className="tiny-input"
              value={threshold}
              readOnly={ignoreScore}
              style={{
                cursor: ignoreScore ? "not-allowed" : "text",
                backgroundColor: ignoreScore ? "#ccc" : "#1f1f25",
                color: "white"
              }}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              title="Minimum cosine similarity score (e.g. 0.8 = 80%)"
            />
            <label title="Include all scores regardless of threshold">
              <input
                type="checkbox"
                checked={ignoreScore}
                onChange={() => setIgnoreScore(!ignoreScore)}
              />
              Ignore Score
            </label>
          </div>

          <button onClick={fetchResults}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        <p className="results-summary">
          {results.length > 0
            ? `${results.length} products matched ${ignoreScore ? "" : `with score ≥ ${(threshold * 100).toFixed(0)}%`}`
            : "No results found. Try a different query."}
        </p>

        <div className={`grid-wrapper ${loading ? 'faded' : ''}`}>
          {loading && (
            <div className="spinner-overlay">
              <div className="spinner"></div>
            </div>
          )}

          <div className="grid">
            {paginatedResults.map((item, idx) => (
              <div key={idx} className="card">
                {item.image_urls && (
                  <img
                    src={item.image_urls.split(",")[0]?.trim()}
                    alt={item.product_title}
                    className="product-image"
                  />
                )}
                <div>
                  <h2 className="product-title">{truncateWords(item.product_title, 10)}</h2>
                  <p className="product-brand"><strong>Brand:</strong> {item.product_brand || "N/A"}</p>
                  <p className={`product-description ${item.product_description ? 'filled' : 'empty'}`}>
                    {item.product_description
                      ? truncateWords(stripHtml(item.product_description), 20)
                      : "No description available."}
                    {item.product_description && (
                      <> <a href="#" className="see-more"> See more</a></>
                    )}
                  </p>
                  <p className={`score ${getScoreClass(item.score)}`}>Score: {(item.score * 100).toFixed(2)}%</p>
                </div>
                <a
                  className="more-btn"
                  href={`/product-detail?data=${encodeURIComponent(JSON.stringify(item))}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  More info
                </a>
              </div>
            ))}
          </div>
        </div>

        {totalPages > 1 && (
          <div className="pagination">
            <button onClick={() => setPage(p => Math.max(p - 1, 1))} disabled={page === 1}>
              Previous
            </button>
            <p>Page {page} of {totalPages}</p>
            <button onClick={() => setPage(p => Math.min(p + 1, totalPages))} disabled={page === totalPages}>
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchBox;
