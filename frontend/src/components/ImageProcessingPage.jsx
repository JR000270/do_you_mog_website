import React, { useEffect, useState } from 'react';
import api from '../api.js';

//uploading image file to be analyzed by the model. submit button sends it
// back to the backend for analysis.
const ImageProcessingPage = () => {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const [analysis, setAnalysis] = useState(null);

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
        <div>
            {/*two columns setup, left for uploaded image and right for the analysis*/}
            <div className="flex flex-row">
                <div className="flex flex-col items-center">
                    {/*if a file has been uploaded display it here */}
                    {selectedFile && (
                        <div className="mt-4">
                            <img
                                src={URL.createObjectURL(selectedFile)}
                                alt="Uploaded Image"
                                className="mt-2 w-75 h-100 object-cover rounded-lg"
                            />
                        </div>
                    )}
                </div>
                <div className="flex flex-col items-center px-10 py-5">
                    {analysis && (
                        <div>
                            <h3 className="text-xl font-semibold text-green-500">Analysis Result:</h3>
                            <p className="text-lg font-medium text-gray-300">{analysis.funny_comment}</p>
                            <p className="text-lg font-medium text-gray-300">Mogging Score: {analysis.mog_probability}</p>
                            <h4 className="text-lg font-medium text-gray-300">Most Mogging Features (Out of 10 points):</h4>
                            <ul className="list-disc list-inside text-gray-300">
                                {Object.entries(analysis.contributions).map(([feature, contribution]) => (
                                    <li key={feature} className="text-gray-300"> {feature}: {contribution} </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </div>
            <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <input type="file" accept="image/*" onChange={handleFileChange} className="p-2 border border-gray-500 rounded mb-4 text-gray-300 cursor-pointer" />
                <button type="submit" disabled={!selectedFile} className="mt-4 px-4 py-2 bg-green-500 text-white font-semibold rounded hover:bg-green-600">Analyze Your Mogging</button>
            </form>
        </div>
    );
};

export default ImageProcessingPage;