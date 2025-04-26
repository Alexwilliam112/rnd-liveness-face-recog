'use client';

import dynamic from 'next/dynamic';

const FaceRecognition = dynamic(() => import('../components/FaceRecognition'), { ssr: false });

export default function Home() {
  return (
    <main style={{ padding: '20px' }}>
      <h1>Face Recognition & Liveness Check</h1>
      <FaceRecognition />
    </main>
  );
}
