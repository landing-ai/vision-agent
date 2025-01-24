"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import React, { useRef, useState, useEffect } from "react";

interface DetectionItem {
  request: {
    prompts: string | string[];
    confidence: number;
    function_name: string;
  };

  response: {
    data: Array<{
      label: string;
      score: number;
      bbox: [number, number, number, number];
    }>;
  };
  files: Array<[string, string]>;
}

interface VisualizerProps {
  detectionResults: DetectionItem[];
  onSubmit?: (functionName: string, boxThreshold: number) => void;
}


const Visualizer: React.FC<VisualizerProps> = ({ detectionResults, onSubmit }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [threshold, setThreshold] = useState(0.05);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % detectionResults.length);
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev - 1 + detectionResults.length) % detectionResults.length);
  };

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setThreshold(parseFloat(e.target.value));
  };

  useEffect(() => {
    if (!detectionResults || detectionResults.length === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const currentResult = detectionResults[currentIndex];

    const image = new Image();
    image.onload = () => {
      canvas.width = image.width;
      canvas.height = image.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);

      currentResult.response.data
        .filter(detection => detection.score >= threshold)
        .forEach(detection => {
          // only handle bbox for now, not segmentation or classification
          if (detection.bbox) {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;

            ctx.strokeStyle = "rgba(255, 0, 0, 0.6)";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);

            ctx.font = "16px Arial";
            const labelText = `${detection.label}: ${detection.score.toFixed(2)}`;
            const textMetrics = ctx.measureText(labelText);
            const textHeight = 20; // Approximate height of the text
            const padding = 4;

            // Draw semi-transparent background for text
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.fillRect(
              x1 - padding,
              y1 - textHeight - padding,
              textMetrics.width + (padding * 2),
              textHeight + (padding * 2)
            );

            // Draw text
            ctx.fillStyle = "white";
            ctx.fillText(labelText, x1, y1 - 5);
          }
      });
    };

    image.src = `data:image/png;base64,${currentResult.files[0][1]}`;
  }, [detectionResults, currentIndex, threshold]);

  if (!detectionResults || detectionResults.length === 0) {
    return <div>No results to visualize</div>;
  }

  const currentResult = detectionResults[currentIndex];

  return (
    <div className="visualizer-container p-4 bg-gray-100 rounded-lg">
      <div className="visualizer-info mb-4">
        <h3 className="text-lg font-bold">Function: {currentResult.request.function_name}</h3>
        <p>Prompt: {Array.isArray(currentResult.request.prompts) 
          ? currentResult.request.prompts.join(', ')
          : currentResult.request.prompts}
        </p>

        <div className="threshold-control mb-4">
          <label htmlFor="threshold-slider" className="block mb-2">
            Confidence Threshold: {threshold.toFixed(2)}
          </label>
          <input
            id="threshold-slider"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={handleThresholdChange}
            className="w-full"
          />
        </div>
      </div>

      <div className="image-navigation-container relative flex items-center justify-center">
        {detectionResults.length > 1 && (
          <button
            onClick={handlePrevious}
            className="absolute left-0 z-10 bg-white/50 rounded-full p-2 hover:bg-white/75"
          >
            <ChevronLeft />
          </button>
        )}

        <canvas ref={canvasRef} className="visualizer-canvas max-w-full" />

        {detectionResults.length > 1 && (
          <button
            onClick={handleNext}
            className="absolute right-0 z-10 bg-white/50 rounded-full p-2 hover:bg-white/75"
          >
            <ChevronRight />
          </button>
        )}
      </div>

      <div className="navigation-info text-center mt-2">
        <p>Image {currentIndex + 1} of {detectionResults.length}</p>
      </div>

      <div className="flex justify-center mt-4">
        <button
          onClick={() => onSubmit?.(currentResult.request.function_name, threshold)}
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded"
        >
          Choose
        </button>
      </div>

    </div>
  );
};


export default Visualizer;
