"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import React, { useRef, useState, useEffect } from "react";

// --- Types ---

interface Detection {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  // Florence2 sometimes returns this property:
  bounding_box?: [number, number, number, number];
  // For images, masks may be available.
  mask?: {
    counts: number[];
    size: number[];
  };
}

interface DetectionItem {
  request: {
    prompts?: string | string[];
    prompt?: string; // qwen2
    confidence: number;
    function_name: string;
  };
  // For images: an array of detections.
  // For video: an array of arrays (each inner array is the detections for a given frame).
  response: {
    data: Detection[] | Detection[][] | string;
  };
  // files[0][0] is either "image" or "video"
  files: Array<[string, string]>;
}

interface VisualizerProps {
  detectionResults: DetectionItem[];
  onSubmit?: (functionName: string, boxThreshold: number) => void;
}

interface VideoVisualizerProps {
  videoSrc: string;              // Base64 video source (with data URI prefix)
  detections: Detection[][];     // Each inner array corresponds to one frame’s detections.
  threshold: number;
  fps?: number;
}


const drawBoundingBox = (ctx: CanvasRenderingContext2D, detection: Detection) => {
  // Use Florence2 compatibility if needed.
  if (detection.bounding_box) {
    detection.bbox = detection.bounding_box;
  }
  if (detection.bbox) {
    const [x1, y1, x2, y2] = detection.bbox;
    const boxWidth = x2 - x1;
    const boxHeight = y2 - y1;

    // Draw bounding box.
    ctx.strokeStyle = "rgba(255, 0, 0, 0.6)";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, boxWidth, boxHeight);

    // Draw label.
    ctx.font = "16px Arial";
    const labelText = `${detection.label}: ${detection.score.toFixed(2)}`;
    const textMetrics = ctx.measureText(labelText);
    const textHeight = 20; // Approximate text height.
    const padding = 4;

    // Draw background behind the text.
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.fillRect(
      x1 - padding,
      y1 - textHeight - padding,
      textMetrics.width + padding * 2,
      textHeight + padding * 2
    );

    // Draw text.
    ctx.fillStyle = "white";
    ctx.fillText(labelText, x1, y1 - 5);
  }
};
  

// --- VideoVisualizer Component ---
// This component renders a <video> with an overlaid canvas.
// On each time update, it computes the current frame index (using the provided fps)
// and draws only bounding boxes (no masks) for detections that meet the threshold.
const VideoVisualizer: React.FC<VideoVisualizerProps> = ({
  videoSrc,
  detections,
  threshold,
  fps = 1,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // When the video metadata loads, set the canvas size to match the video.
  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;

    const handleLoadedMetadata = () => {
      if (canvasRef.current) {
        canvasRef.current.width = videoEl.videoWidth;
        canvasRef.current.height = videoEl.videoHeight;
      }
    };

    videoEl.addEventListener("loadedmetadata", handleLoadedMetadata);
    return () => videoEl.removeEventListener("loadedmetadata", handleLoadedMetadata);
  }, []);

  // On each time update, clear the canvas and draw bounding boxes for the current frame.
  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;

    const handleTimeUpdate = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Clear canvas for the new frame.
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Calculate the current frame (assumes constant fps).
      const currentFrame = Math.floor(videoEl.currentTime * fps);
      const frameDetections = detections[currentFrame] || [];

      frameDetections
        .filter((det) => det.score >= threshold)
        .forEach((detection) => {
          drawBoundingBox(ctx, detection);
        });
    };

    videoEl.addEventListener("timeupdate", handleTimeUpdate);
    return () => videoEl.removeEventListener("timeupdate", handleTimeUpdate);
  }, [detections, threshold, fps]);

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <video
        ref={videoRef}
        src={videoSrc}
        controls
        className="max-w-full rounded-lg"
      />
      <canvas
        ref={canvasRef}
        className="visualizer-canvas"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
    </div>
  );
};

// --- VisualizerHiL Component ---
// For image results, we use your existing canvas drawing (which also draws masks).
// For video results, we render the VideoVisualizer component (which draws only boxes).
const VisualizerHiL: React.FC<VisualizerProps> = ({ detectionResults, onSubmit }) => {
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

  const currentResult = detectionResults[currentIndex];

  // Only run the image drawing effect if this result is an image.
  useEffect(() => {
    if (!detectionResults || detectionResults.length === 0) return;
    if (currentResult.files[0][0] === "video") return; // Skip image drawing for video.

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = new Image();
    image.onload = () => {
      canvas.width = image.width;
      canvas.height = image.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);

      if (typeof currentResult.response.data === 'string') {
        // If response is a string, write it on the image
        ctx.font = "64px Arial";
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, 10, canvas.width - 20, 80);
        ctx.fillStyle = "white";
        ctx.fillText(currentResult.response.data, 20, 70);
      } else {
      
        // For images, assume currentResult.response.data is an array of detections.
        (currentResult.response.data as Detection[])
          .filter((detection) => detection.score >= threshold)
          .forEach((detection) => {
            // Draw mask if available (only for images).
            if (detection.mask && detection.mask.counts && detection.mask.size) {
              const [height, width] = detection.mask.size;
              const counts = detection.mask.counts;

              const tempCanvas = document.createElement("canvas");
              tempCanvas.width = width;
              tempCanvas.height = height;
              const tmpCtx = tempCanvas.getContext("2d");
              if (!tmpCtx) return;

              const bitmap = new Uint8Array(width * height);
              let pixelIndex = 0;
              let isOne = false;

              for (const count of counts) {
                for (let i = 0; i < count; i++) {
                  if (pixelIndex < bitmap.length) {
                    // Convert from row-major to column-major order.
                    const x = Math.floor(pixelIndex / height);
                    const y = pixelIndex % height;
                    const newIndex = y * width + x;
                    if (newIndex < bitmap.length) {
                      bitmap[newIndex] = isOne ? 1 : 0;
                    }
                    pixelIndex++;
                  }
                }
                isOne = !isOne;
              }

              const imageData = tmpCtx.createImageData(width, height);
              for (let i = 0; i < bitmap.length; i++) {
                const offset = i * 4;
                if (bitmap[i] === 1) {
                  imageData.data[offset] = 255;
                  imageData.data[offset + 1] = 0;
                  imageData.data[offset + 2] = 0;
                  imageData.data[offset + 3] = 170;
                }
              }
              tmpCtx.putImageData(imageData, 0, 0);

              ctx.save();
              ctx.globalCompositeOperation = "source-over";
              ctx.drawImage(tempCanvas, 0, 0, width, height);
              ctx.restore();
            }

            drawBoundingBox(ctx, detection);
          });
      }
    };

    image.src = `data:image/png;base64,${currentResult.files[0][1]}`;
  }, [detectionResults, currentIndex, threshold]);

  return (
    <div className="visualizer-container p-4 bg-gray-100 rounded-lg">
      <div className="visualizer-info mb-4">
        <h3 className="text-lg font-bold">
          Function: {currentResult.request.function_name}
        </h3>
        <p>
          Prompt:{" "}
          {Array.isArray(currentResult.request.prompts)
            ? currentResult.request.prompts.join(", ")
            : currentResult.request.prompts
            ? currentResult.request.prompts
            : currentResult.request.prompt}
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

        {currentResult.files[0][0] === "video" ? (
          <VideoVisualizer
            videoSrc={`data:video/mp4;base64,${currentResult.files[0][1]}`}
            // Cast detection data to Detection[][] (make sure your backend returns the video detections in this format)
            detections={currentResult.response.data as Detection[][]}
            threshold={threshold}
            fps={1}
          />
        ) : (
          <canvas ref={canvasRef} className="visualizer-canvas max-w-full" />
        )}

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
        <p>
          Image {currentIndex + 1} of {detectionResults.length}
        </p>
      </div>

      <div className="flex justify-center mt-4">
        <button
          onClick={() =>
            onSubmit?.(currentResult.request.function_name, threshold)
          }
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded"
        >
          Choose
        </button>
      </div>
    </div>
  );
};

export { VisualizerHiL };
