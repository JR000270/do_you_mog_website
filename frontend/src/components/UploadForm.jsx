import React, { useEffect, useState } from 'react';
import api from '../api.js';

//uploading image file to be analyzed by the model. submit button sends it
// back to the backend for analysis.
const UploadForm = () => {
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
            <h2>Upload Your Image</h2>
            <div>
                {analysis && (
                    <div>
                        <h3>Analysis Result:</h3>
                        <p>Mog Probability: {analysis.mog_probability}</p>
                        <h4>Feature Contributions:</h4>
                        <ul>
                            {Object.entries(analysis.contributions).map(([feature, contribution]) => (
                                <li key={feature}> {feature}: {contribution.toFixed(2)} </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
            <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <button type="submit" disabled={!selectedFile}>Analyze Your Mogging</button>
            </form>
        </div>
    );
};

export default UploadForm;