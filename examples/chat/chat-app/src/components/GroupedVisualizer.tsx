"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, ChevronDown } from "lucide-react";
import { Detection, DetectionItem } from "./types";
import { ImageVisualizer } from "./ImageVisualizer"; // Your image canvas component
import { VideoVisualizer } from "./VideoVisualizer"; // Your video visualizer component


// --- Group Type ---
// Each group contains all items that share the same function_name.
interface GroupedDetection {
  functionName: string;
  items: DetectionItem[];
}

interface VisualizerProps {
  detectionResults: DetectionItem[];
  onSubmit?: (functionName: string, boxThreshold: number) => void;
}

// --- GroupedVisualizer Component ---

const GroupedVisualizer: React.FC<VisualizerProps> = ({
  detectionResults,
  onSubmit,
}) => {
  // 1. Group detectionResults by function_name.
  const groups: GroupedDetection[] = useMemo(() => {
    const groupMap: Record<string, DetectionItem[]> = {};
    detectionResults.forEach((item) => {
      const fn = item.request.function_name;
      if (!groupMap[fn]) {
        groupMap[fn] = [];
      }
      groupMap[fn].push(item);
    });
    return Object.entries(groupMap).map(([functionName, items]) => ({
      functionName,
      items,
    }));
  }, [detectionResults]);

  // 2. Maintain state for the currently active group (across different function_names)
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);

  // 3. Maintain state for the currently selected image index within each group.
  //    The key is the group’s function name.
  const [selectedIndices, setSelectedIndices] = useState<Record<string, number>>(
    {}
  );

  // When groups change, initialize the selectedIndices for each group to zero.
  useEffect(() => {
    const initialIndices: Record<string, number> = {};
    groups.forEach((group) => {
      initialIndices[group.functionName] = 0;
    });
    setSelectedIndices(initialIndices);
  }, [groups]);

  // 4. Global threshold state.
  const [threshold, setThreshold] = useState(0.05);

  // 5. Determine the current group and current item.
  const currentGroup = groups[currentGroupIndex];
  const currentItem =
    currentGroup.items[selectedIndices[currentGroup.functionName] ?? 0];

  // 6. Navigation handlers:

  // For switching groups (different function_names)
  const handlePreviousGroup = () => {
    setCurrentGroupIndex((prev) => (prev - 1 + groups.length) % groups.length);
  };

  const handleNextGroup = () => {
    setCurrentGroupIndex((prev) => (prev + 1) % groups.length);
  };

  // For cycling images within the same group.
  const handleNextImageInGroup = () => {
    setSelectedIndices((prev) => {
      const currentIndex = prev[currentGroup.functionName] ?? 0;
      const nextIndex = (currentIndex + 1) % currentGroup.items.length;
      return { ...prev, [currentGroup.functionName]: nextIndex };
    });
  };

  return (
    <div className="grouped-visualizer p-4 bg-gray-100 rounded-lg">
      {/* Group Info and Threshold */}
      <div className="visualizer-info mb-4">
        <h3 className="text-lg font-bold">
          Function: {currentGroup.functionName}
        </h3>
        <p>
          Prompt:{" "}
          {Array.isArray(currentItem.request.prompts)
            ? currentItem.request.prompts.join(", ")
            : currentItem.request.prompts || currentItem.request.prompt}
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
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Visualization Area */}
      <div className="visualizer-navigation relative flex items-center justify-center">
        {/* Left/Right Buttons: Navigate between groups (different function_names) */}
        {groups.length > 1 && (
          <>
            <button
              onClick={handlePreviousGroup}
              className="absolute left-2 z-10 bg-white/50 rounded-full p-2 hover:bg-white/75"
              title="Previous group"
            >
              <ChevronLeft />
            </button>
            <button
              onClick={handleNextGroup}
              className="absolute right-2 z-10 bg-white/50 rounded-full p-2 hover:bg-white/75"
              title="Next group"
            >
              <ChevronRight />
            </button>
          </>
        )}

        {/* Render image or video visualizer */}
        {currentItem.files[0][0] === "video" ? (
          <VideoVisualizer
            videoSrc={`data:video/mp4;base64,${currentItem.files[0][1]}`}
            // Ensure your backend returns video detections in the proper format (Detection[][])
            detections={currentItem.response.data as Detection[][]}
            threshold={threshold}
            fps={1}
          />
        ) : (
          <ImageVisualizer detectionItem={currentItem} threshold={threshold} />
        )}

        {/* Down Arrow: Cycle within images/videos of the same group */}
        {currentGroup.items.length > 1 && (
          <button
            onClick={handleNextImageInGroup}
            className="absolute bottom-2 left-1/2 transform -translate-x-1/2 z-10 bg-white/50 rounded-full p-2 hover:bg-white/75"
            title="Next image in group"
          >
            <ChevronDown />
          </button>
        )}
      </div>

      {/* Navigation Info */}
      <div className="navigation-info text-center mt-2">
        <p>
          Tool {currentGroupIndex + 1} of {groups.length} &mdash; Tool Media{" "}
          {(selectedIndices[currentGroup.functionName] ?? 0) + 1} of{" "}
          {currentGroup.items.length}
        </p>
      </div>

      {/* Submit/Choose Button */}
      <div className="flex justify-center mt-4">
        <button
          onClick={() =>
            onSubmit?.(currentItem.request.function_name, threshold)
          }
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded"
        >
          Choose
        </button>
      </div>
    </div>
  );
};

export { GroupedVisualizer };
