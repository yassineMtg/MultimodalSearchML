import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchBox from './components/SearchBox';
import ProductDetail from './components/ProductDetail';
import ABTestingPage from './components/ABTestingPage';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SearchBox />} />
        <Route path="/product-detail" element={<ProductDetail />} />
        <Route path="/ab-test" element={<ABTestingPage />} />
      </Routes>
    </Router>
  );
};

export default App;
