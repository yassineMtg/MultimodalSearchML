import React, { useState } from 'react';
import axios from "axios";
import './SearchBox.scss';

const SearchBox = () => {
    const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", {
        query,
        k: 5,
      });
      setResults(response.data.results);
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4 md:px-10">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-center">🔍 Smart Product Search</h1>

        <div className="flex items-center gap-4 mb-8">
          <input
            type="text"
            className="flex-grow border border-gray-300 p-3 rounded-xl shadow-sm focus:outline-none focus:ring focus:border-blue-500"
            placeholder="Search for a product (e.g. wireless headphones)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            onClick={handleSearch}
            className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 transition"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {results.length > 0 && (
          <div className="grid gap-6">
            {results.map((item, idx) => (
              <div key={idx} className="p-4 border border-gray-200 rounded-xl shadow bg-white">
                <div className="flex gap-4 items-center">
                  {item.image_urls && (
                    <img
                      src={item.image_urls.split(",")[0]?.trim()}
                      alt={item.product_title}
                      className="w-24 h-24 object-contain rounded"
                    />
                  )}
                  <div>
                    <h2 className="text-lg font-semibold">{item.product_title}</h2>
                    <p className="text-gray-500 text-sm mb-2">
                      <span className="font-medium">Brand:</span> {item.product_brand || "N/A"} • <span className="font-medium">Color:</span> {item.product_color || "N/A"}
                    </p>
                    <p className="text-sm text-gray-600">{item.product_description || "No description available."}</p>
                    <p className="mt-2 text-sm text-blue-600 font-semibold">Score: {item.score.toFixed(4) * 100}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchBox;
