import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import api from '../api.js';

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


    return (
        <div className="py-10">
            {/*two columns setup, left for uploaded image and right for the analysis*/}
            <div className="flex flex-row">
                <div className="flex flex-col items-center">
                    {/*if a file has been uploaded display it here */}
                    {selectedFile && (
                        <motion.div className="mt-4" initial={{scaleY: 0}} animate={{scaleY: 1, x: submitted ? 0 : 70}} transition={{duration:0.5}}>
                            {/*If no analysis in yet -> potential mogger, otherwise based on probability -> mog or not mog */}
                            <h2 className={'text-2xl font-semibold ' + (!analysis ? "text-yellow-500" : analysis && analysis.mog_probability >= 50 ? "text-green-500" : "text-red-500")}>
                            { 
                                !analysis ? "Potential Mogger:" :
                                    analysis && analysis.mog_probability >= 50 ? "Certified Mogger:" : "Not a Mogger:"
                            }
                            </h2>
                            <img
                                src={URL.createObjectURL(selectedFile)}
                                alt="Uploaded Image"
                                className="mt-2 w-75 h-100 object-cover rounded-lg"
                            />
                        </motion.div>
                    )}
                </div>
                <div className="flex flex-col items-center px-10 py-7">
                    {analysis && (
                        <motion.div initial={{x: -200, y: 25}} animate={{x:0}} transition={{duration:0.5}}>
                            <p className="text-lg font-medium text-gray-300">{analysis.funny_comment}</p>
                            <p className="text-lg font-medium text-gray-300">Mogging Score: {analysis.mog_probability}</p>
                            <h4 className="text-lg font-medium text-gray-300">Features Ratings:</h4>
                            <ul className="list-disc list-inside text-gray-300">
                                {Object.entries(analysis.contributions).map(([feature, contribution]) => (
                                    <li key={feature} className="text-gray-300"> {feature}: {contribution}/10 </li>
                                ))}
                            </ul>
                        </motion.div>
                    )}
                </div>
            </div>
            
            <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <input type="file" accept="image/*" onChange={handleFileChange} className="p-2 border text-yellow-500 rounded mb-4 border-yellow-500 cursor-pointer " />
                <button onClick={() => setSubmitted(true)} type="submit" disabled={!selectedFile} className="mt-4 px-4 py-2 bg-yellow-500 text-white font-semibold rounded hover:bg-yellow-600">Analyze</button>
            </form>
            
        </div>
    );
};

export default ImageProcessingPage;