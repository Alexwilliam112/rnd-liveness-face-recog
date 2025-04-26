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

export const performLivenessAndRecognition = async (
  videoRef,
  canvasRef,
  referenceDescriptor,
  updateProgressMessage,
  onComplete
) => {
  try {
    const options = new faceapi.TinyFaceDetectorOptions();
    let smileDetected = false;

    const MATCH_RATE_THRESHOLD = 70; // Lowered match rate threshold to 70

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

        // Check for smile
        const isSmile = expressions.happy > 0.2;

        if (!smileDetected) {
          updateProgressMessage('Please smile.');
        }

        if (isSmile && !smileDetected) {
          updateProgressMessage('Smile detected! Proceeding to face recognition...');
          const faceMatcher = new faceapi.FaceMatcher(referenceDescriptor, 0.6);
          const match = faceMatcher.findBestMatch(detection.descriptor);
          const matchRate = ((1 - match.distance) * 100).toFixed(2);

          if (matchRate >= MATCH_RATE_THRESHOLD) {
            smileDetected = true;
            updateProgressMessage(`Smile SUCCESS: Match Rate ${matchRate}%`);
          } else {
            updateProgressMessage(`Smile FAILED: Match Rate ${matchRate}%`);
          }
        }

        // If smile detection and recognition are successful, complete the process
        if (smileDetected) {
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