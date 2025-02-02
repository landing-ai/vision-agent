import { Detection } from "./types";
import { useRef, useEffect } from "react";
import { drawBoundingBox } from "./utils";


interface VideoVisualizerProps {
  videoSrc: string;              // Base64 video source (with data URI prefix)
  detections: Detection[][];     // Each inner array corresponds to one frame’s detections.
  threshold: number;
  fps?: number;
}


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
        loop
        autoPlay
        muted // Required for autoplay to work in most browsers
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

export { VideoVisualizer };
