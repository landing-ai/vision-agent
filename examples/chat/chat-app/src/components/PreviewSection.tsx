"use client";

import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { gruvboxLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { PolygonDrawer, Polygon } from "@/components/PolygonDrawer";
import { ResultImageWithBoundingBoxes } from "@/components/ResultImageWithBoundingBoxes";

interface PreviewSectionProps {
  uploadedMedia: string | null;
  uploadedFile: string | null;
  uploadedResult: number[][] | null;
  onPolygonsChange: (polygons: Polygon[]) => void;
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

interface File {
  name: string;
  content: string;
  type: "code" | "image";
}

export function PreviewSection({
  uploadedMedia,
  uploadedFile,
  uploadedResult,
  onPolygonsChange,
  activeTab = "media",
  onTabChange,
}: PreviewSectionProps) {
  return (
    <Card className="overflow-y-auto h-[800px]">
      <Tabs value={activeTab} onValueChange={onTabChange}>
        <TabsList className="w-full justify-start rounded-none bg-gray-50">
          <TabsTrigger value="media" className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
              <circle cx="9" cy="9" r="2" />
              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
            </svg>
            Media
          </TabsTrigger>
          <TabsTrigger value="code" className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
            Code
          </TabsTrigger>
          <TabsTrigger value="result" className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 3H5a2 2 0 0 0-2 2v14c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-3" />
              <path d="M18 3h3v3" />
              <path d="M21 13V6h-8" />
              <path d="m16 8-8 8" />
            </svg>
            Result
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="media" className="p-4 bg-white flex-1 flex flex-col">
          <ScrollArea className="flex-1">
            <div className="border rounded-md p-4 bg-gray-50 shadow-inner">
              {uploadedMedia ? (
                <PolygonDrawer
                  media={uploadedMedia || ""}
                  onPolygonsChange={onPolygonsChange}
                />
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-gray-300 mb-4">
                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                    <circle cx="9" cy="9" r="2" />
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                  </svg>
                  <p className="text-gray-500">No media uploaded yet.</p>
                  <p className="text-sm text-gray-400 mt-2">Upload media to begin annotation.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </TabsContent>
        
        <TabsContent value="code" className="p-4 bg-white flex-1 flex flex-col">
          <ScrollArea className="flex-1">
            <div className="mb-4">
              {uploadedFile ? (
                <SyntaxHighlighter
                  language="python"
                  style={gruvboxLight}
                  customStyle={{
                    padding: "1.25rem",
                    borderRadius: "0.5rem",
                    backgroundColor: "#f8f9fa",
                    fontSize: "0.875rem",
                    border: "1px solid #e2e8f0",
                    boxShadow: "inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)"
                  }}
                  wrapLongLines={true}
                >
                  {uploadedFile || ""}
                </SyntaxHighlighter>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center border rounded-md bg-gray-50">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-gray-300 mb-4">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  <p className="text-gray-500">No code uploaded yet.</p>
                  <p className="text-sm text-gray-400 mt-2">Ask VisionAgent a question and wait for it to generate code.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </TabsContent>
        
        <TabsContent value="result" className="p-4 bg-white">
          <div className="border rounded-md p-4 bg-gray-50 shadow-inner">
            {uploadedResult && uploadedMedia ? (
              <ResultImageWithBoundingBoxes
                imageSrc={uploadedMedia}
                boundingBoxes={uploadedResult}
              />
            ) : uploadedResult ? (
              <div className="text-center p-4">
                <p className="text-yellow-600 mb-2">Bounding box coordinates detected but no original image available.</p>
                <div className="bg-gray-100 p-3 rounded text-sm font-mono">
                  {JSON.stringify(uploadedResult, null, 2)}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-gray-300 mb-4">
                  <path d="M8 3H5a2 2 0 0 0-2 2v14c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-3" />
                  <path d="M18 3h3v3" />
                  <path d="M21 13V6h-8" />
                  <path d="m16 8-8 8" />
                </svg>
                <p className="text-gray-500">No result uploaded yet.</p>
                <p className="text-sm text-gray-400 mt-2">Results will appear here after processing.</p>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
}
