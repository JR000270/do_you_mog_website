import React from 'react';
import './App.css';
import ImageProcessingPage from './components/ImageProcessingPage';
import { motion } from "motion/react"

const App = () => {
  return (
    <div className="main-background min-h-screen flex flex-col items-center py-15 relative overflow-hidden">
      <div className="rays" aria-hidden="true" />
      <div className="dome" aria-hidden="true" />
      <header className="relative z-10">
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
      <motion.main className="relative z-10" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.5, duration: 2 }}>
        <ImageProcessingPage />
      </motion.main>
    </div>
  );
};

export default App;