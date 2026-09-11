import React, { useRef } from "react";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "user"
};

// react-webcam's screenshot comes back as a base64 data URL; the rest of the
// app (preview <img>, FormData upload) expects a File, so convert it here.
function dataURLtoFile(dataUrl, filename) {
    const [header, base64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new File([bytes], filename, { type: mime });
}

export default function WebCam_Modal({ isOpen, onClose, onCapture }) {
    const webcamRef = useRef(null);

    if (!isOpen) return null;

    const handleCapture = () => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;
        const file = dataURLtoFile(imageSrc, `webcam-${Date.now()}.jpeg`);
        onCapture(file);
        onClose();
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/10 backdrop-blur-none p-4 animate-fadeIn"
            onClick={onClose}
        >
            {/* stop propagation so clicking the webcam/buttons doesn't bubble to the backdrop and close the modal */}
            <div className="bg-gray-900 p-4 rounded-lg flex flex-col items-center gap-4" onClick={(e) => e.stopPropagation()}>
                <Webcam
                    audio={false}
                    height={720}
                    screenshotFormat="image/jpeg"
                    width={1280}
                    videoConstraints={videoConstraints}
                    ref={webcamRef}
                    className="rounded-lg"
                />
                <div className="flex gap-4">
                    <button
                        type="button"
                        onClick={handleCapture}
                        className="px-4 py-2 bg-yellow-500 text-white font-semibold rounded hover:bg-yellow-600"
                    >
                        Capture photo
                    </button>
                    <button
                        type="button"
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-700 text-white font-semibold rounded hover:bg-gray-600"
                    >
                        Cancel
                    </button>
                </div>
            </div>
        </div>
    );
}