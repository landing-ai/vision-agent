"use client";

import { useState } from "react";
import { ChatSection } from "@/components/ChatSection";
import { PreviewSection } from "@/components/PreviewSection";
import { VersionSelector } from "@/components/VersionSelector";
import { Polygon } from "@/components/PolygonDrawer";

export default function Component() {
  const [selectedVersion, setSelectedVersion] = useState<'v2' | 'v3' | null>(null);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const handleFileUpload = (file: string) => setUploadedFile(file);

  const [uploadedImage, setUploadedMedia] = useState<string | null>(null);
  const handleMediaUpload = (image: string) => setUploadedMedia(image);

  const [uploadedResult, setUploadedResult] = useState<number[][] | null>(null);
  const handleResultUpload = (result: number[][]) => setUploadedResult(result);

  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const handlePolygonChange = (polygons: Polygon[]) => setPolygons(polygons);

  const [activeTab, setActiveTab] = useState<string>("media");
  const handleTabChange = (tab: string) => setActiveTab(tab);

  const handleVersionSelect = (version: 'v2' | 'v3' | null) => {
    setSelectedVersion(version);
  };

  return (
    <>
      {!selectedVersion && (
        <VersionSelector onVersionSelect={handleVersionSelect} />
      )}
      <div className="h-screen grid grid-cols-2 gap-4 p-4 bg-background">
        <ChatSection
          uploadedMedia={uploadedImage}
          onUploadedMedia={handleMediaUpload}
          uploadedFile={uploadedFile}
          onUploadedFile={handleFileUpload}
          uploadedResult={uploadedResult}
          onUploadedResult={handleResultUpload}
          polygons={polygons}
          version={selectedVersion}
          onTabChange={handleTabChange}
        />
        <PreviewSection
          uploadedMedia={uploadedImage}
          uploadedFile={uploadedFile}
          uploadedResult={uploadedResult}
          onPolygonsChange={handlePolygonChange}
          activeTab={activeTab}
          onTabChange={handleTabChange}
        />
      </div>
    </>
  );
}
