import * as faceapi from 'face-api.js';

// Load face-api.js models
export const loadModels = async (MODEL_URL, updateProgressMessage) => {
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    ]);
    updateProgressMessage('Models loaded successfully.');
    return true;
  } catch (error) {
    console.error('Error loading models:', error);
    updateProgressMessage('Error loading models.');
    return false;
  }
};

// Process reference image
export const processReferenceImage = async (file, updateProgressMessage) => {
  try {
    const img = await faceapi.bufferToImage(file);
    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection) {
      const descriptor = new faceapi.LabeledFaceDescriptors('Reference', [detection.descriptor]);
      updateProgressMessage('Reference image processed successfully.');
      return descriptor;
    } else {
      updateProgressMessage('No face detected in the uploaded image.');
      return null;
    }
  } catch (error) {
    console.error('Error processing reference image:', error);
    updateProgressMessage('Failed to process the uploaded image.');
    return null;
  }
};

// Perform face recognition and liveness check
export const performFaceRecognition = async (
  videoRef,
  canvasRef,
  referenceDescriptor,
  updateProgressMessage
) => {
  try {
    const options = new faceapi.TinyFaceDetectorOptions();

    updateProgressMessage('Detecting faces...');
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
      updateProgressMessage('Analyzing expressions for liveness...');
      const expressions = detections[0].expressions;
      const isLive = expressions.happy > 0.7 || expressions.surprised > 0.7;

      updateProgressMessage('Matching face with reference image...');
      const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
      const match = faceMatcher.findBestMatch(detections[0].descriptor);
      const isMatch = match.label === 'Reference';
      const matchRate = ((1 - match.distance) * 100).toFixed(2);

      return {
        isLive,
        isMatch,
        matchRate,
      };
    } else {
      updateProgressMessage('No faces detected.');
      return null;
    }
  } catch (error) {
    console.error('Error during face recognition and liveness check:', error);
    updateProgressMessage('An error occurred during the check.');
    return null;
  }
};

export const performLivenessAndRecognition = async (
  videoRef,
  canvasRef,
  referenceDescriptor,
  updateProgressMessage,
  onComplete
) => {
  try {
    const options = new faceapi.TinyFaceDetectorOptions();
    let blinkDetected = false;
    let smileDetected = false;

    // Adjustable EAR threshold for blink detection
    const BLINK_THRESHOLD = 0.6; // Increased from 0.2 to 0.25 for better sensitivity

    const processFrame = async () => {
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
        const detection = detections[0];
        const expressions = detection.expressions;

        // Check for blink (based on eye landmarks)
        const landmarks = detection.landmarks;
        const leftEye = landmarks.getLeftEye();
        const rightEye = landmarks.getRightEye();
        const eyeAspectRatio = (eye) => {
          const width = Math.hypot(eye[3].x - eye[0].x, eye[3].y - eye[0].y);
          const height = Math.hypot(eye[1].x - eye[5].x, eye[1].y - eye[5].y);
          return height / width;
        };

        const leftEAR = eyeAspectRatio(leftEye);
        const rightEAR = eyeAspectRatio(rightEye);

        // Log EAR values for debugging
        updateProgressMessage(`Left EAR: ${leftEAR.toFixed(3)}, Right EAR: ${rightEAR.toFixed(3)}`);

        const isBlink = leftEAR < BLINK_THRESHOLD && rightEAR < BLINK_THRESHOLD;

        if (!blinkDetected) {
          updateProgressMessage('Please blink.');
        }

        if (isBlink && !blinkDetected) {
          updateProgressMessage('Blink detected! Proceeding to face recognition...');
          const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
          const match = faceMatcher.findBestMatch(detection.descriptor);
          const matchRate = ((1 - match.distance) * 100).toFixed(2);

          if (matchRate >= 80) {
            blinkDetected = true;
            updateProgressMessage(`Blink SUCCESS: Match Rate ${matchRate}%`);
          } else {
            updateProgressMessage(`Blink FAILED: Match Rate ${matchRate}%`);
          }
        }

        // Check for smile
        const isSmile = expressions.happy > 0.7;

        if (blinkDetected && !smileDetected) {
          updateProgressMessage('Please smile.');
        }

        if (isSmile && blinkDetected && !smileDetected) {
          updateProgressMessage('Smile detected! Proceeding to face recognition...');
          const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
          const match = faceMatcher.findBestMatch(detection.descriptor);
          const matchRate = ((1 - match.distance) * 100).toFixed(2);

          if (matchRate >= 80) {
            smileDetected = true;
            updateProgressMessage(`Smile SUCCESS: Match Rate ${matchRate}%`);
          } else {
            updateProgressMessage(`Smile FAILED: Match Rate ${matchRate}%`);
          }
        }

        // If both criteria are met, complete the process
        if (blinkDetected && smileDetected) {
          updateProgressMessage('✅ Liveness and Face Recognition checks completed successfully!');
          // Ensure the success message is logged before completing
          setTimeout(onComplete, 500); // Add a slight delay to ensure the message is visible
          return;
        }
      } else {
        updateProgressMessage('No faces detected. Please ensure your face is visible to the camera.');
      }

      // Continue processing frames
      requestAnimationFrame(processFrame);
    };

    // Start processing frames
    processFrame();
  } catch (error) {
    console.error('Error during liveness and face recognition check:', error);
    updateProgressMessage('An error occurred during the check.');
  }
};