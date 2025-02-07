"use client";

import React, { useEffect, useRef } from "react";
import { Detection, DetectionItem } from "./types";
import { drawBoundingBox } from "./utils";

// (Re-use your Detection and DetectionItem types and drawBoundingBox function here)

interface ImageVisualizerProps {
  detectionItem: DetectionItem;
  threshold: number;
}

const ImageVisualizer: React.FC<ImageVisualizerProps> = ({
  detectionItem,
  threshold,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Skip if this is a video.
    if (detectionItem.files[0][0] === "video") return;
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

      if (typeof detectionItem.response.data === "string") {
        // Draw response string (for text-based responses).
        const fontSize = Math.min(canvas.width, canvas.height) * 0.05;
        ctx.font = `${fontSize}px Arial`;
        
        // Text wrapping configuration
        const maxWidth = canvas.width - 40; // Padding on both sides
        const lineHeight = fontSize * 1.2;
        const padding = 20;
        
        // Wrap text into lines
        const words = detectionItem.response.data.split(' ');
        const lines: string[] = [];
        let currentLine = words[0];
        
        for (let i = 1; i < words.length; i++) {
            const testLine = currentLine + ' ' + words[i];
            const metrics = ctx.measureText(testLine);
            if (metrics.width > maxWidth) {
                lines.push(currentLine);
                currentLine = words[i];
            } else {
                currentLine = testLine;
            }
        }
        lines.push(currentLine);
        
        // Calculate background height based on number of lines
        const bgHeight = (lines.length * lineHeight) + (padding * 2);
        
        // Draw background
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, 10, canvas.width - 20, bgHeight);
        
        // Draw text lines
        ctx.fillStyle = "white";
        lines.forEach((line, i) => {
            ctx.fillText(line, padding, padding + (i + 1) * lineHeight);
        });
      } else if (Array.isArray(detectionItem.response.data)) {
        // For images, assume response.data is an array of Detection.
        (detectionItem.response.data as Detection[])
          .filter((detection) => detection.score >= threshold)
          .forEach((detection) => {
            // Draw mask if available (only for images).
            if (
              detection.mask &&
              detection.mask.counts &&
              detection.mask.size
            ) {
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

            // (If needed, draw the mask here as in your original code.)
            drawBoundingBox(ctx, detection);
          });
      }
    };
    image.src = `data:image/png;base64,${detectionItem.files[0][1]}`;
  }, [detectionItem, threshold]);

  return <canvas ref={canvasRef} className="visualizer-canvas max-w-full" />;
};

export { ImageVisualizer };
