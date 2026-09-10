import React from 'react';
import './App.css';
import ImageProcessingPage from './components/ImageProcessingPage';
import { motion } from "motion/react"

const App = () => {
  return (
    <div className="bg-gray-900 min-h-screen flex flex-col items-center py-15">
      <header>
        <motion.h1
          initial="hidden"
          animate="visible"
          variants={{
            visible: { transition: { staggerChildren: 0.7 } },
          }}
          className="text-4xl font-bold text-yellow-500 flex gap-20"
        >
          {["DO", "YOU", "MOG?"].map((word) => (
            <motion.span
              key={word}
              className="inline-block px-5 text-center"
              variants={{
                hidden: { scale: 0, y: 800 },
                visible: { 
                  scale: [0,10,2], 
                  y: 0, 
                  transition: { duration: 1.5 },
                   rotate: [0, 25, -25, 0]
                  },
              }}
            >
              {word}
            </motion.span>
          ))}
        </motion.h1>
      </header>
      <main>
        <ImageProcessingPage />
      </main>
    </div>
  );
};

export default App;