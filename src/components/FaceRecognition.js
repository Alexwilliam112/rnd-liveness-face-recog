'use client';

import React, { useRef, useEffect, useState } from 'react';
import { loadModels, processReferenceImage, performLivenessAndRecognition } from '../utils/faceRecognitionUtils';

const FaceRecognition = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [referenceDescriptor, setReferenceDescriptor] = useState(null);
  const [cameraError, setCameraError] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');

  const updateProgressMessage = (message) => {
    setProgressMessage(message);
  };

  useEffect(() => {
    const MODEL_URL = '/models';
    loadModels(MODEL_URL, updateProgressMessage).then((loaded) => setModelsLoaded(loaded));
  }, []);

  useEffect(() => {
    if (!modelsLoaded || typeof window === 'undefined' || !showCamera) return;

    updateProgressMessage('Starting camera...');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setCameraReady(true);
            updateProgressMessage('Camera is ready.');
          };
        }
      })
      .catch((err) => {
        console.error('Error accessing webcam:', err);
        setCameraError(true);
        updateProgressMessage('Error accessing webcam.');
      });
  }, [modelsLoaded, showCamera]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const descriptor = await processReferenceImage(file, updateProgressMessage);
    setReferenceDescriptor(descriptor);

    // Automatically start the camera after a successful image upload
    if (descriptor) {
      setShowCamera(true);
      setCameraReady(false); // Reset cameraReady to false until the camera is ready
    }
  };

  const handleCheck = () => {
    if (!modelsLoaded || !referenceDescriptor || !cameraReady) {
      updateProgressMessage('Ensure models are loaded, reference image is uploaded, and camera is ready.');
      return;
    }

    setIsChecking(true);
    updateProgressMessage('Starting liveness and face recognition check...');

    performLivenessAndRecognition(videoRef, canvasRef, referenceDescriptor, updateProgressMessage, () => {
      setIsChecking(false);
    });
  };

  if (cameraError) {
    return <div style={{ fontSize: '20px', color: 'red', textAlign: 'center', marginTop: '20px' }}>Require Camera</div>;
  }

  return (
    <div>
      <h2>Upload Reference Image</h2>
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button
          onClick={() => {
            setShowCamera(true);
            handleCheck();
          }}
          disabled={!referenceDescriptor || isChecking || !cameraReady}

          style={{
            marginTop: '20px',
            padding: '10px 20px',
            fontSize: '16px',

            cursor: referenceDescriptor && !isChecking && cameraReady ? 'pointer' : 'not-allowed',
            backgroundColor: referenceDescriptor && !isChecking && cameraReady ? '#007BFF' : '#CCC',

            color: '#FFF',
            border: 'none',
            borderRadius: '5px',
          }}
        >
          {isChecking
            ? 'Checking...'

            : !cameraReady
            ? 'LOADING CAMERA...'

            : 'Start Liveness + Face Recognition'}
        </button>

        <div style={{ marginLeft: '20px', textAlign: 'left', fontSize: '16px' }}>
          <p>Progress Log:</p>
          <p>{progressMessage}</p>
        </div>
      </div>

      {showCamera && (
        <div style={{ position: 'relative', width: '720px', height: '560px', marginTop: '20px' }}>
          <video
            ref={videoRef}
            width="720"
            height="560"
            autoPlay
            muted
            style={{ position: 'absolute', top: 0, left: 0 }}
          />
          <canvas
            ref={canvasRef}
            width="720"
            height="560"
            style={{ position: 'absolute', top: 0, left: 0 }}
          />
        </div>
      )}
    </div>
  );
};

export default FaceRecognition;