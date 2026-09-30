import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import api from '../api.js';
import WebCam_Modal from './WebCam_Modal.jsx';

//uploading image file to be analyzed by the model. submit button sends it
// back to the backend for analysis.
const ImageProcessingPage = () => {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const [analysis, setAnalysis] = useState(null);
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit =  async () => {
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            const response = await api.post('/analyze', formData);
            setAnalysis(response.data);
        }catch (error) {
            console.error('Error analyzing the image:', error);
        }
    };


    //webcam modal state management
    const [isModalOpen, setIsModalOpen] = useState(false);
    //open webcam modal when the user clicks the button
    const openWebcamModal = () => {
        setIsModalOpen(true);
    }
    //called by the modal once the user captures a photo
    const handleCapture = (file) => {
        setSelectedFile(file);
    }


    return (
        <div className="w-full py-10 flex flex-col items-center">
            {/*stacked on phones, two columns (image left, analysis right) from md (768px) up*/}
            <div className="flex flex-col md:flex-row items-center md:items-start w-full">
                <div className="flex flex-col items-center">
                    {/*if a file has been uploaded display it here */}
                    {selectedFile && (
                        <motion.div className="mt-4"
                          key={selectedFile.name + selectedFile.lastModified}
                          style={{ transformPerspective: 1000 }}
                          initial="hidden"
                          animate="visible"
                          variants={{
                            hidden: {scale: 0, rotateY: 0},
                            visible:
                            {
                                scale: [0, 1.7, 1.8, 1.7, 1],
                                rotateY: [0, 720, 720, 720],
                                transition: {duration: 3, times: [0, 0.5, 0.83, 1], ease: "easeInOut"},
                            }

                        }}>
                            {/*If no analysis in yet -> potential mogger, otherwise based on probability -> mog or not mog */}
                            {/*key change on analysis arrival remounts the h2, retriggering the pop-in animation*/}
                            <motion.h2
                              key={analysis ? 'result' : 'pending'}
                              initial={{ scale: 0, opacity: 0, rotate: -8 }}
                              animate={{ scale: [0, 10, 0.9, 1], opacity: 1, rotate: 0 }}
                              transition={{ duration: 2, ease: "easeOut" }}
                           
                              className={'text-2xl font-display font-semibold ' + (!analysis ? "text-purple-500" : analysis && analysis.mog_probability >= 50 ? "text-green-500" : "text-red-500")}>
                            {
                                !analysis ? "Potential Mogger:" :
                                    analysis && analysis.mog_probability >= 50 ? "YOU CERTAINLY DO!" : "YOU DO NOT!"
                            }
                            </motion.h2>
                            <img
                                src={URL.createObjectURL(selectedFile)}
                                alt="Uploaded Image"
                                className="mt-2 w-full max-w-75 aspect-3/4 object-cover rounded-lg"
                            />
                        </motion.div>
                    )}
                </div>
                <div className="flex flex-col items-center w-full md:w-auto px-0 md:px-10 py-7">
                    {analysis && (
                        <motion.div initial={{x: -200, y: 25}} animate={{x:0}} transition={{duration:0.5}}
                        className="backdrop:blur-sm bg-black/30 border border-purple-500 p-4 rounded-lg">
                            <p className="text-lg font-display font-medium text-gray-300">{analysis.funny_comment}</p>
                            <p className="text-lg font-display font-medium text-gray-300">Score: {analysis.mog_probability}</p>
                            <h4 className="text-lg font-display font-medium text-gray-300">Features Ratings:</h4>
                            <ul className="font-display list-disc list-inside text-gray-300">
                                {Object.entries(analysis.contributions).map(([feature, contribution]) => (
                                    <li key={feature} className="text-gray-300"> {feature}: {contribution}/10 </li>
                                ))}
                            </ul>
                        </motion.div>
                    )}
                </div>
            </div>
            
            {( !analysis && (
            <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
                {/* Open webcam to upload photo or select from device */}
                <div className="flex flex-col sm:flex-row items-center gap-4 mb-4 py-10">
                    <input type="file" accept="image/*" onChange={handleFileChange} className="w-full sm:w-auto p-2 font-display backdrop:blur-sm bg-black/30 border text-purple-500 rounded border-purple-500 hover:bg-purple-500 hover:text-white cursor-pointer " />
                    <button type="button" onClick={openWebcamModal} className="px-4 py-2 font-display backdrop:blur-sm bg-black/30 border text-purple-500 rounded border-purple-500 cursor-pointer hover:bg-purple-500 hover:text-white">Take a Photo</button>
                </div>
                <button onClick={() => setSubmitted(true)} type="submit" disabled={!selectedFile} className="mt-4 px-4 py-2 bg-purple-500 text-white font-display rounded hover:bg-purple-600">Analyze</button>
            </form>
            ))} 
            {( submitted && !analysis &&(
                <div className="flex items-center justify-center py-16">
                    <motion.h2
                        className="text-4xl md:text-6xl font-display font-bold text-purple-400 select-none"
                        animate={{
                            y: [-5, -18, -5, 18, -5],
                            rotate: [-7, 0, 7, 0, -7],
                        }}
                        transition={{
                            y: { duration: 2.4, repeat: Infinity, ease: "easeInOut" },
                            rotate: { duration: 3.2, repeat: Infinity, ease: "easeInOut" },
                        }}
                    >
                        Scanning for mogging...
                    </motion.h2>
                </div>
            ))}
            {(analysis &&(
                <button onClick={() => {setSubmitted(false); setSelectedFile(null); setAnalysis(null);}} type="button" className="mt-4 px-4 py-2 bg-purple-500 text-white font-display rounded hover:bg-purple-600">Try Another!</button>
            ))}

            <WebCam_Modal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onCapture={handleCapture}
            />

        </div>
    );
};

export default ImageProcessingPage;