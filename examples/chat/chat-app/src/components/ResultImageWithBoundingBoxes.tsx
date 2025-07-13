"use client";

import { useEffect, useRef } from "react";
import { drawBoundingBox } from "./utils";
import { Detection } from "./types";

interface ResultImageWithBoundingBoxesProps {
  imageSrc: string;
  boundingBoxes: number[][];
}

export function ResultImageWithBoundingBoxes({
  imageSrc,
  boundingBoxes,
}: ResultImageWithBoundingBoxesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image || !boundingBoxes?.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const drawImageWithBoundingBoxes = () => {
      // Set canvas dimensions to match image
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      
      // Clear canvas and draw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);

      // Convert coordinate arrays to Detection objects and draw bounding boxes
      boundingBoxes.forEach((coords, index) => {
        if (coords.length >= 4) {
          // Unnormalize coordinates from 0-1 range to pixel coordinates
          const x1 = coords[0] * canvas.width;
          const y1 = coords[1] * canvas.height;
          const x2 = coords[2] * canvas.width;
          const y2 = coords[3] * canvas.height;
          
          const detection: Detection = {
            bbox: [x1, y1, x2, y2],
            label: `Detection ${index + 1}`,
            score: 1.0,
          };
          drawBoundingBox(ctx, detection);
        }
      });
    };

    // Draw when image loads
    if (image.complete && image.naturalHeight !== 0) {
      drawImageWithBoundingBoxes();
    } else {
      image.onload = drawImageWithBoundingBoxes;
    }
  }, [imageSrc, boundingBoxes]);

  return (
    <div className="relative">
      <img
        ref={imageRef}
        src={imageSrc}
        alt="Original"
        className="max-w-full rounded-md border shadow-sm opacity-0 absolute"
        crossOrigin="anonymous"
      />
      <canvas
        ref={canvasRef}
        className="max-w-full rounded-md border shadow-sm"
        style={{
          maxHeight: "600px",
          objectFit: "contain",
        }}
      />
    </div>
  );
}
