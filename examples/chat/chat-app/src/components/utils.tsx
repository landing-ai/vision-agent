import { Detection } from "./types";


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
  
export { drawBoundingBox };
