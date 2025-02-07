"use client";

import React, { useEffect, useState, useRef, MouseEvent } from "react";
import { Button } from "@/components/ui/button";

interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  id: number;
  name: string;
  points: Point[];
}

interface PolygonDrawerProps {
  media: string;
  onPolygonsChange?: (polygons: Polygon[]) => void;
}

const PolygonDrawer: React.FC<PolygonDrawerProps> = ({
  media,
  onPolygonsChange,
}) => {
  // State to store saved polygons
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  // State to store points for the polygon currently being drawn
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [intrinsicDimensions, setIntrinsicDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });

  // Ref for the SVG overlay (to compute mouse coordinates correctly)
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mediaRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);
  const mediaType = media.startsWith("data:video/") ? "video" : "image";

  useEffect(() => {
    if (onPolygonsChange) {
      onPolygonsChange(polygons);
    }
  }, [polygons, onPolygonsChange]);

  /**
   * Called when the media has loaded its metadata (video) or loaded (image)
   * so that we can capture its intrinsic dimensions.
   */
  const handleMediaLoad = () => {
    if (!mediaRef.current) return;
    if (mediaType === "video") {
      const video = mediaRef.current as HTMLVideoElement;
      setIntrinsicDimensions({
        width: video.videoWidth,
        height: video.videoHeight,
      });
    } else {
      const image = mediaRef.current as HTMLImageElement;
      setIntrinsicDimensions({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    }
  };

  /**
   * Returns the intrinsic dimensions of the media.
   */
  const getIntrinsicDimensions = (): { width: number; height: number } => {
    return intrinsicDimensions;
  };

  /**
   * Converts a point from displayed (SVG) coordinates into intrinsic coordinates.
   */
  const getIntrinsicPoint = (displayedX: number, displayedY: number): Point => {
    if (!svgRef.current) return { x: displayedX, y: displayedY };
    const svgRect = svgRef.current.getBoundingClientRect();
    const { width: intrinsicWidth, height: intrinsicHeight } =
      getIntrinsicDimensions();
    if (
      svgRect.width === 0 ||
      svgRect.height === 0 ||
      intrinsicWidth === 0 ||
      intrinsicHeight === 0
    ) {
      return { x: displayedX, y: displayedY };
    }
    const scaleX = intrinsicWidth / svgRect.width;
    const scaleY = intrinsicHeight / svgRect.height;
    return { x: displayedX * scaleX, y: displayedY * scaleY };
  };

  /**
   * Converts a point from intrinsic coordinates into displayed (SVG) coordinates.
   */
  const getDisplayedPoint = (intrinsicX: number, intrinsicY: number): Point => {
    if (!svgRef.current) return { x: intrinsicX, y: intrinsicY };
    const svgRect = svgRef.current.getBoundingClientRect();
    const { width: intrinsicWidth, height: intrinsicHeight } =
      getIntrinsicDimensions();
    if (intrinsicWidth === 0 || intrinsicHeight === 0) {
      return { x: intrinsicX, y: intrinsicY };
    }
    const scaleX = svgRect.width / intrinsicWidth;
    const scaleY = svgRect.height / intrinsicHeight;
    return { x: intrinsicX * scaleX, y: intrinsicY * scaleY };
  };

  // Handler for clicks on the SVG overlay.
  // Adds a point (vertex) to the current polygon.
  const handleSvgClick = (e: MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const displayedX = e.clientX - rect.left;
    const displayedY = e.clientY - rect.top;
    const intrinsicPoint = getIntrinsicPoint(displayedX, displayedY);
    setCurrentPoints((prev) => [...prev, intrinsicPoint]);
  };

  // Completes the current polygon by prompting for a name.
  // If valid, the polygon is saved and the onPolygonAdded callback is fired.
  const handleFinishPolygon = () => {
    if (currentPoints.length < 3) {
      alert("A polygon must have at least 3 points.");
      return;
    }
    const name = prompt("Enter a name for the polygon:");
    if (!name) return;

    const newPolygon: Polygon = {
      id: Date.now(), // For production, consider a more robust id generation
      name,
      points: currentPoints,
    };

    console.log("New polygon:", newPolygon);
    setPolygons((prevPolygons) => [...prevPolygons, newPolygon]);
    setCurrentPoints([]);
  };

  // Deletes a polygon by its id and fires the onPolygonRemoved callback.
  const handleDeletePolygon = (id: number) => {
    setPolygons((prevPolygons) => prevPolygons.filter((p) => p.id !== id));
  };

  /**
   * Converts an array of intrinsic points into a string for the SVG "points" attribute.
   */
  const convertPointsToString = (points: Point[]): string => {
    if (!svgRef.current) {
      return points.map((p) => `${p.x},${p.y}`).join(" ");
    }
    return points
      .map((p) => {
        const displayed = getDisplayedPoint(p.x, p.y);
        return `${displayed.x},${displayed.y}`;
      })
      .join(" ");
  };

  return (
    <div>
      {/* Container for the video and the SVG overlay */}
      <div style={{ position: "relative", display: "inline-block" }}>
        {media ? (
          mediaType === "video" ? (
            <video
              ref={mediaRef as React.RefObject<HTMLVideoElement>}
              src={media}
              controls
              loop
              autoPlay
              muted
              onLoadedMetadata={handleMediaLoad}
              style={{ display: "block" }}
            />
          ) : (
            <img
              ref={mediaRef as React.RefObject<HTMLImageElement>}
              src={media}
              onLoad={handleMediaLoad}
              style={{ display: "block" }}
              alt="Drawable"
            />
          )
        ) : (
          <p>No media uploaded yet.</p>
        )}
        {/* SVG overlay for drawing polygons */}
        <svg
          ref={svgRef}
          onClick={handleSvgClick}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            cursor: "crosshair",
          }}
        >
          {/* Render saved polygons */}
          {polygons.map((polygon) => (
            <polygon
              key={polygon.id}
              points={convertPointsToString(polygon.points)}
              fill="rgba(0, 128, 255, 0.3)"
              stroke="blue"
              strokeWidth="2"
              onClick={(e) => {
                // Prevent the click from also adding a new point.
                e.stopPropagation();
                if (window.confirm(`Delete polygon "${polygon.name}"?`)) {
                  handleDeletePolygon(polygon.id);
                }
              }}
            />
          ))}
          {/* Render the polygon being drawn (as an open polyline) */}
          {currentPoints.length > 0 && (
            <polyline
              points={convertPointsToString(currentPoints)}
              fill="none"
              stroke="red"
              strokeWidth="2"
            />
          )}
        </svg>
      </div>

      {/* Controls for finishing or clearing the current polygon */}
      <div style={{ marginTop: "10px" }}>
        <Button onClick={handleFinishPolygon} className="mr-2">
          Add Polygon
        </Button>
        <Button onClick={() => setCurrentPoints([])}>Undo</Button>
      </div>

      {/* List of saved polygons with delete buttons */}
      <div className="mt-6 p-4 border rounded-lg bg-white shadow-sm">
        <h3 className="text-lg font-semibold mb-3">Saved Polygons</h3>
        {polygons.length === 0 ? (
          <p className="text-gray-500 italic">No polygons saved yet.</p>
        ) : (
          <ul className="space-y-2">
            {polygons.map((polygon) => (
              <li
                key={polygon.id}
                className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-md"
              >
                <span className="font-medium">{polygon.name}</span>
                <Button
                  onClick={() => handleDeletePolygon(polygon.id)}
                  variant="destructive"
                  size="sm"
                >
                  Delete
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export { PolygonDrawer };
