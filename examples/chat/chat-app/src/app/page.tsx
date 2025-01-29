"use client";

import { useState } from "react";
import { ChatSection } from "@/components/ChatSection";
import { PreviewSection } from "@/components/PreviewSection";

export default function Component() {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const handleFileUpload = (file: string) => setUploadedFile(file);

  const [uploadedImage, setUploadedMedia] = useState<string | null>(null);
  const handleMediaUpload = (image: string) => setUploadedMedia(image);

  const [uploadedResult, setUploadedResult] = useState<string | null>(null);
  const handleResultUpload = (result: string) => setUploadedResult(result);

  return (
    <div className="h-screen grid grid-cols-2 gap-4 p-4 bg-background">
      <ChatSection
        uploadedMedia={uploadedImage}
        onUploadedMedia={handleMediaUpload}
        uploadedFile={uploadedFile}
        onUploadedFile={handleFileUpload}
        uploadedResult={uploadedResult}
        onUploadedResult={handleResultUpload}
      />
      <PreviewSection
        uploadedMedia={uploadedImage}
        uploadedFile={uploadedFile}
        uploadedResult={uploadedResult}
      />
    </div>
  );
}
