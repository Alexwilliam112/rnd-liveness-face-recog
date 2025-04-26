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
  const [showCamera, setShowCamera] = useState(false);
  const [cameraReady, setCameraReady] = useState(false); // New state to track camera readiness
  const [progressMessages, setProgressMessages] = useState([]); // Log of all progress messages

  // Helper function to add a message to the progress log
  const addProgressMessage = (message) => {
    setProgressMessages((prevMessages) => [...prevMessages, message]);
  };

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = '/models';
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL), // Ensure SsdMobilenetv1 is loaded
        ]);
        setModelsLoaded(true);
        addProgressMessage('Models loaded successfully.');
      } catch (error) {
        console.error('Error loading models:', error);
        addProgressMessage('Error loading models.');
      }
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
          setCameraReady(true); // Set camera as ready
          addProgressMessage('Camera started successfully.');
        })
        .catch((err) => {
          console.error('Error accessing webcam:', err);
          setCameraError(true);
          addProgressMessage('Error accessing webcam.');
        });
    } else {
      console.error('getUserMedia is not supported in this browser.');
      setCameraError(true);
      addProgressMessage('getUserMedia is not supported in this browser.');
    }
  }, [modelsLoaded, showCamera]);

  // Upload and process reference image
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const img = await faceapi.bufferToImage(file);
      const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        const descriptor = new faceapi.LabeledFaceDescriptors('Reference', [detection.descriptor]);
        setReferenceDescriptor(descriptor);
        addProgressMessage('Reference image processed successfully.');
      } else {
        alert('No face detected in the uploaded image.');
        addProgressMessage('No face detected in the uploaded image.');
      }
    } catch (error) {
      console.error('Error processing reference image:', error);
      alert('Failed to process the uploaded image. Please try again.');
      addProgressMessage('Failed to process the uploaded image.');
    }
  };

  // Face recognition & liveness check
  const handleCheck = async () => {
    console.log('Button Clicked');

    if (!modelsLoaded) {
      alert('Models are still loading. Please wait and try again.');
      addProgressMessage('Models are still loading. Please wait and try again.');
      return;
    }

    if (!referenceDescriptor) {
      alert('Please upload a reference image before starting the check.');
      addProgressMessage('Please upload a reference image before starting the check.');
      return;
    }

    if (!cameraReady) {
      alert('Camera is not ready yet. Please wait.');
      addProgressMessage('Camera is not ready yet. Please wait.');
      return;
    }

    setIsChecking(true);
    addProgressMessage('Starting liveness and face recognition check...');

    try {
      const options = new faceapi.TinyFaceDetectorOptions();

      addProgressMessage('Detecting faces...');
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
        addProgressMessage('Analyzing expressions for liveness...');
        const expressions = detections[0].expressions;
        const isLive = expressions.happy > 0.7 || expressions.surprised > 0.7;
        setLivenessPassed(isLive);

        addProgressMessage('Matching face with reference image...');
        const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
        const match = faceMatcher.findBestMatch(detections[0].descriptor);
        const isMatch = match.label === 'Reference';

        setMatchResult(isMatch ? 'PASS' : 'FAILED');

        // Update progress message based on results
        if (isMatch && isLive) {
          addProgressMessage('✅ Both Face Recognition and Liveness Check Passed!');
        } else if (isMatch) {
          addProgressMessage('✅ Face Recognition Passed, ❌ Liveness Check Failed.');
        } else if (isLive) {
          addProgressMessage('❌ Face Recognition Failed, ✅ Liveness Check Passed.');
        } else {
          addProgressMessage('❌ Both Face Recognition and Liveness Check Failed.');
        }
      } else {
        addProgressMessage('No faces detected.');
        setMatchResult(null);
        setLivenessPassed(false);
      }
    } catch (error) {
      console.error('Error during face recognition and liveness check:', error);
      alert('An error occurred during the check. Please try again.');
      addProgressMessage('An error occurred during the check.');
    } finally {
      setIsChecking(false);
    }
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

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
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

        <div style={{ marginLeft: '20px', textAlign: 'left', fontSize: '16px' }}>
          <p>Progress Log:</p>
          <ul>
            {progressMessages.map((msg, index) => (
              <li key={index}>{msg}</li>
            ))}
          </ul>
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