import React from 'react';
import './App.css';
import ImageProcessingPage from './components/ImageProcessingPage';

const App = () => {
  return (
    <div className="bg-gray-900 min-h-screen flex flex-col items-center py-15">
      <header>
        <h1 className="text-4xl font-bold text-green-500">Do You Mog?</h1>
      </header>
      <main>
        <ImageProcessingPage />
      </main>
    </div>
  );
};

export default App;