'use client';

import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';

const FaceRecognition = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [referenceDescriptor, setReferenceDescriptor] = useState(null);
  const [matchResult, setMatchResult] = useState(null);
  const [livenessPassed, setLivenessPassed] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [showCamera, setShowCamera] = useState(false); // State to control camera visibility

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = '/models';
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      ]);
      setModelsLoaded(true);
    };
    loadModels();
  }, []);

  // Start webcam (client-side only)
  useEffect(() => {
    if (!modelsLoaded || typeof window === 'undefined' || !showCamera) return;

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.error('Error accessing webcam:', err);
          setCameraError(true);
        });
    } else {
      console.error('getUserMedia is not supported in this browser.');
      setCameraError(true);
    }
  }, [modelsLoaded, showCamera]);

  // Upload and process reference image
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = await faceapi.bufferToImage(file);
    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection) {
      setReferenceDescriptor(
        new faceapi.LabeledFaceDescriptors('Reference', [detection.descriptor])
      );
    } else {
      alert('No face detected in the uploaded image.');
    }
  };

  // Face recognition & liveness check
  const handleCheck = async () => {
    if (!modelsLoaded || !referenceDescriptor) return;

    setShowCamera(true); // Show the camera frame
    setIsChecking(true); // Indicate that the check is running

    const options = new faceapi.TinyFaceDetectorOptions();

    const detections = await faceapi
      .detectAllFaces(videoRef.current, options)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors();

    if (canvasRef.current && videoRef.current) {
      const dims = faceapi.matchDimensions(canvasRef.current, videoRef.current, true);
      canvasRef.current.getContext('2d').clearRect(0, 0, dims.width, dims.height);

      const resized = faceapi.resizeResults(detections, dims);
      faceapi.draw.drawDetections(canvasRef.current, resized);
      faceapi.draw.drawFaceLandmarks(canvasRef.current, resized);
      faceapi.draw.drawFaceExpressions(canvasRef.current, resized);
    }

    if (detections.length > 0) {
      const expressions = detections[0].expressions;
      const isLive = expressions.happy > 0.7 || expressions.surprised > 0.7;
      setLivenessPassed(isLive);

      const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
      const match = faceMatcher.findBestMatch(detections[0].descriptor);
      setMatchResult(match.label === 'Reference' ? 'PASS' : 'FAILED');
    } else {
      setMatchResult(null);
      setLivenessPassed(false);
    }

    setIsChecking(false); // Indicate that the check is complete
  };

  if (cameraError) {
    return (
      <div style={{ fontSize: '20px', color: 'red', textAlign: 'center', marginTop: '20px' }}>
        Require Camera
      </div>
    );
  }

  return (
    <div>
      <h2>Upload Reference Image</h2>
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      <button
        onClick={handleCheck}
        disabled={!referenceDescriptor || isChecking}
        style={{
          marginTop: '20px',
          padding: '10px 20px',
          fontSize: '16px',
          cursor: referenceDescriptor && !isChecking ? 'pointer' : 'not-allowed',
          backgroundColor: referenceDescriptor && !isChecking ? '#007BFF' : '#CCC',
          color: '#FFF',
          border: 'none',
          borderRadius: '5px',
        }}
      >
        {isChecking ? 'Checking...' : 'Start Liveness + Face Recognition'}
      </button>

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

      <div style={{ marginTop: '20px', fontSize: '20px' }}>
        <p>
          Face Recognition:{' '}
          <strong style={{ color: matchResult === 'PASS' ? 'green' : 'red' }}>
            {matchResult ? (matchResult === 'PASS' ? '✅ PASS' : '❌ FAILED') : '...'}
          </strong>
        </p>

        <p>
          Liveness Check:{' '}
          <strong style={{ color: livenessPassed ? 'green' : 'red' }}>
            {livenessPassed ? '✅ PASS' : '❌ FAILED'}
          </strong>
        </p>
      </div>
    </div>
  );
};

export default FaceRecognition;