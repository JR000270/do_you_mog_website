import React from 'react';
import './App.css';
import UploadForm from './components/UploadForm';

const App = () => {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Do You Mog?</h1>
      </header>
      <main>
        <UploadForm />
      </main>
    </div>
  );
};

export default App;