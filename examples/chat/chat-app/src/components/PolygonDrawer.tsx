"use client";

import React, { useEffect, useState, useRef, MouseEvent } from "react";
import { Button } from "@/components/ui/button";
import { PlusCircle, Undo2, Trash2, Menu, X } from "lucide-react";

interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  id: number;
  name: string;
  points: Point[];
}

const adjectiveToColor: Record<string, string> = {
  Red: "rgba(255, 0, 0, 0.3)",
  Blue: "rgba(0, 0, 255, 0.3)",
  Green: "rgba(0, 128, 0, 0.3)",
  Purple: "rgba(128, 0, 128, 0.3)",
  Golden: "rgba(255, 215, 0, 0.3)",
  Silver: "rgba(192, 192, 192, 0.3)",
  Crystal: "rgba(173, 216, 230, 0.3)", // Light blue-ish
  Shadow: "rgba(50, 50, 50, 0.3)",
  Bright: "rgba(255, 255, 0, 0.3)", // Yellow
};

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
  // State to store the current preview point (mouse position)
  const [previewPoint, setPreviewPoint] = useState<Point | null>(null);
  const [intrinsicDimensions, setIntrinsicDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  
  // State for menu visibility
  const [isMenuVisible, setIsMenuVisible] = useState(true);

  // Ref for the SVG overlay (to compute mouse coordinates correctly)
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mediaRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);
  const mediaType = media.startsWith("data:video/") ? "video" : "image";

  useEffect(() => {
    if (onPolygonsChange) {
      onPolygonsChange(polygons);
    }
  }, [polygons, onPolygonsChange]);

  const getPolygonColor = (name: string): string => {
    const firstWord = name.split(" ")[0]; // Assumes adjective is first
    return adjectiveToColor[firstWord] || "rgba(0, 128, 255, 0.3)"; // default blue-ish
  };  
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

  // Handler for mouse move events over the SVG.
  // Updates the preview point for the polygon being drawn.
  const handleSvgMouseMove = (e: MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const displayedX = e.clientX - rect.left;
    const displayedY = e.clientY - rect.top;
    const intrinsicPoint = getIntrinsicPoint(displayedX, displayedY);
    setPreviewPoint(intrinsicPoint);
  };

  // When the mouse leaves the SVG, clear the preview point so that the polygon auto-connects.
  const handleSvgMouseLeave = () => {
    setPreviewPoint(null);
  };

  // Completes the current polygon by prompting for a name.
  // If valid, the polygon is saved.
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
    setPreviewPoint(null);
  };

  // Deletes a polygon by its id.
  const handleDeletePolygon = (id: number) => {
    setPolygons((prevPolygons) => prevPolygons.filter((p) => p.id !== id));
  };

  // Reset the current drawing
  const handleClearPoints = () => {
    setCurrentPoints([]);
    setPreviewPoint(null);
  };

  // Function to generate random polygon
  const generateRandomPolygon = () => {
    // Get dimensions of the media container
    if (!svgRef.current) return;
    const svgRect = svgRef.current.getBoundingClientRect();
    const width = svgRect.width;
    const height = svgRect.height;
    
    // Generate between 3-8 random points
    const numPoints = Math.floor(Math.random() * 6) + 3;
    const points: Point[] = [];
    
    // Create points that form a somewhat realistic polygon
    // Start with a center point
    const centerX = width / 2;
    const centerY = height / 2;

    // Ensure the center point is within bounds
    const radius = Math.min(width, height) / (Math.random() * 2 + 3);
    
    const offsetX = Math.random() * width - radius / 2;
    const offsetY = Math.random() * height - radius / 2;
    
    // Generate points in a rough circle around the center
    for (let i = 0; i < numPoints; i++) {
      // Use angle to distribute points around the center
      const angle = (i / numPoints) * 2 * Math.PI;
      // Add some randomness to the radius
      const randomRadius = radius * (0.5 + Math.random() * 0.5);
      
      // Calculate coordinates
      const x = offsetX + randomRadius * Math.cos(angle);
      const y = offsetY + randomRadius * Math.sin(angle);
      
      // Convert to intrinsic coordinates
      points.push(getIntrinsicPoint(x, y));
    }
    
    // Generate a random name
    const shapes = ["Triangle", "Quadrilateral", "Pentagon", "Hexagon", "Heptagon", "Octagon"];
    const adjectives = ["Red", "Blue", "Green", "Purple", "Golden", "Silver", "Crystal", "Shadow", "Bright"];
    const name = `${adjectives[Math.floor(Math.random() * adjectives.length)]} ${shapes[numPoints - 3]}`;
    
    // Create and save the polygon
    const newPolygon: Polygon = {
      id: Date.now(),
      name,
      points,
    };
    
    setPolygons((prevPolygons) => [...prevPolygons, newPolygon]);
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

  // Toggle menu visibility
  const toggleMenu = () => {
    setIsMenuVisible(!isMenuVisible);
  };

  return (
    <div className="relative flex flex-col h-full">
      {/* Main content container with native scrolling */}
      <div className="overflow-y-auto h-full">
        <div className="p-1">
          {/* Container for the media and the SVG overlay - CENTER THE MEDIA */}
          <div className="flex justify-center">
            <div style={{ position: "relative", display: "inline-block", maxHeight: "330px" }}>
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
                    style={{ 
                      display: "block", 
                      maxHeight: "330px",
                      width: "auto",
                      height: "auto",
                      objectFit: "contain"
                    }}
                  />
                ) : (
                  <img
                    ref={mediaRef as React.RefObject<HTMLImageElement>}
                    src={media}
                    onLoad={handleMediaLoad}
                    style={{ 
                      display: "block", 
                      maxHeight: "330px",
                      width: "auto",
                      height: "auto",
                      objectFit: "contain"
                    }}
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
                onMouseMove={handleSvgMouseMove}
                onMouseLeave={handleSvgMouseLeave}
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
                    fill={getPolygonColor(polygon.name)}
                    stroke="black"            
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
                {/* Render the shape being drawn */}
                {currentPoints.length > 0 &&
                  (currentPoints.length < 3 ? (
                    // For fewer than 3 points, show a red polyline (with preview if available)
                    <polyline
                      points={convertPointsToString(
                        previewPoint ? [...currentPoints, previewPoint] : currentPoints
                      )}
                      fill="none"
                      stroke="red"
                      strokeWidth="2"
                    />
                  ) : (
                    // For 3 or more points, render a blue filled polygon.
                    <polygon
                      points={convertPointsToString(
                        previewPoint ? [...currentPoints, previewPoint] : currentPoints
                      )}
                      fill="rgba(0, 128, 255, 0.3)"
                      stroke="blue"
                      strokeWidth="2"
                    />
                  ))}
              </svg>
            </div>
          </div>
  
          {/* Enhanced Saved Polygons Section */}
          <div className="mt-6 border rounded-lg bg-white shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-4 sticky top-0 z-10">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-white flex items-center">
                  Saved Polygons <span className="ml-2 bg-white text-purple-600 text-xs rounded-full px-2 py-0.5">{polygons.length}</span>
                </h3>
                
                {polygons.length > 0 && (
                  <Button 
                    onClick={() => setPolygons([])} 
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-1.5 bg-red-500 text-white hover:bg-red-600 border-red-400"
                  >
                    <Trash2 size={14} />
                    <span>Clear All</span>
                  </Button>
                )}
              </div>
            </div>
            
            {polygons.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-4 bg-gray-50">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                  <svg xmlns="http://www.w3.org/3300/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
                    <path d="M12 4L4 8l8 4 8-4-8-4z"/>
                    <path d="M4 12l8 4 8-4"/>
                    <path d="M4 16l8 4 8-4"/>
                  </svg>
                </div>
                <p className="text-gray-600 font-medium text-center">No polygons saved yet</p>
                <p className="text-sm text-gray-500 mt-1 text-center max-w-xs">Draw points on the image to create polygons or use the "Generate Random" button</p>
              </div>
            ) : (
              <div className="max-h-60 overflow-y-auto">
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-4">
                  {polygons.map((polygon) => (
                    <li
                      key={polygon.id}
                      className="group relative flex items-center gap-3 p-3 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-100 rounded-md hover:shadow-md transition-all duration-330"
                    >
                      <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-sm">
                        {polygon.points.length}
                      </div>
                      <div className="flex-grow">
                        <span className="font-medium text-gray-800 text-sm">{polygon.name}</span>
                        <div className="text-xs text-blue-600">
                          {polygon.points.length} {polygon.points.length === 1 ? 'vertex' : 'vertices'}
                        </div>
                      </div>
                      <div className="absolute right-1 top-1 opacity-0 group-hover:opacity-100 transition-opacity duration-330">
                        <Button
                          onClick={() => handleDeletePolygon(polygon.id)}
                          variant="ghost"
                          size="sm"
                          className="flex-shrink-0 text-red-500 hover:bg-red-50 hover:text-red-600 rounded-full p-1 h-6 w-6"
                        >
                          <Trash2 size={14} />
                        </Button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
  
      {/* Floating Menu Button - Always visible */}
      <div className="fixed bottom-6 right-6 z-20">
        <button
          onClick={toggleMenu}
          className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-330"
          title={isMenuVisible ? "Hide Menu" : "Show Menu"}
        >
          {isMenuVisible ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
  
      {/* Floating Action Controls - Collapsible */}
      {isMenuVisible && (
        <div className="fixed bottom-20 right-6 flex flex-col items-end space-y-3 z-10">
          {/* Point counter badge - only shown when drawing */}
          {currentPoints.length > 0 && (
            <div className="bg-gray-800 text-white text-xs font-medium px-2.5 py-1 rounded-full flex items-center">
              <span className="mr-1.5 bg-blue-500 rounded-full w-2 h-2"></span>
              {currentPoints.length} point{currentPoints.length !== 1 ? 's' : ''}
            </div>
          )}
          
          {/* Controls card */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
            {/* Add polygon button - only enabled if there are 3+ points */}
            <button
              onClick={handleFinishPolygon}
              disabled={currentPoints.length < 3}
              className={`w-full flex items-center gap-3 px-5 py-3 transition-colors duration-330 border-b border-gray-100
                ${currentPoints.length >= 3 
                  ? "text-blue-600 hover:bg-blue-50" 
                  : "text-gray-400 cursor-not-allowed"}`}
            >
              <div className={`p-2 rounded-full ${currentPoints.length >= 3 ? "bg-blue-100" : "bg-gray-100"}`}>
                <PlusCircle size={18} className={currentPoints.length >= 3 ? "text-blue-600" : "text-gray-400"} />
              </div>
              <span className="font-medium">Complete Polygon</span>
            </button>
            
            {/* Undo button - only enabled if there are points */}
            <button
              onClick={handleClearPoints}
              disabled={currentPoints.length === 0}
              className={`w-full flex items-center gap-3 px-5 py-3 transition-colors duration-330
                ${currentPoints.length > 0 
                  ? "text-gray-700 hover:bg-gray-50" 
                  : "text-gray-400 cursor-not-allowed"}`}
            >
              <div className={`p-2 rounded-full ${currentPoints.length > 0 ? "bg-gray-100" : "bg-gray-50"}`}>
                <Undo2 size={18} className={currentPoints.length > 0 ? "text-gray-600" : "text-gray-400"} />
              </div>
              <span className="font-medium">Clear Points</span>
            </button>
            
            {/* Generate Random button */}
            <button
              onClick={generateRandomPolygon}
              className="w-full flex items-center gap-3 px-5 py-3 text-purple-600 hover:bg-purple-50 transition-colors duration-330 border-t border-gray-100"
            >
              <div className="bg-purple-100 p-2 rounded-full">
                <svg xmlns="http://www.w3.org/3300/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600">
                  <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
                  <path d="M12 12v9"></path>
                  <path d="m16 16-4-4-4 4"></path>
                </svg>
              </div>
              <span className="font-medium">Generate Random</span>
            </button>
          </div>
          
          {/* Quick add - floating action button - only if 3+ points */}
          {currentPoints.length >= 3 && (
            <button
              onClick={handleFinishPolygon}
              className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-330"
              title="Add Polygon"
            >
              <PlusCircle size={24} />
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export { PolygonDrawer };
