export interface Detection {
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
}''

export interface DetectionItem {
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
};


