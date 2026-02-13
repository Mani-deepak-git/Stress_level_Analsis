import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import InterviewRoom from './pages/InterviewRoom';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/interview/:roomId" element={<InterviewRoom />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;