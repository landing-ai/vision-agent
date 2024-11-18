"use client";

import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { gruvboxLight } from "react-syntax-highlighter/dist/esm/styles/prism";

interface PreviewSectionProps {
  uploadedImage: string | null;
  uploadedFile: string | null;
  uploadedResult: string | null;
}

interface File {
  name: string;
  content: string;
  type: "code" | "image";
}

export function PreviewSection({
  uploadedImage,
  uploadedFile,
  uploadedResult,
}: PreviewSectionProps) {
  return (
    <Card className="overflow-hidden">
      <Tabs defaultValue="media">
        <TabsList className="w-full justify-start border-b rounded-none">
          <TabsTrigger value="media">Media</TabsTrigger>
          <TabsTrigger value="code">Code</TabsTrigger>
          <TabsTrigger value="result">Result</TabsTrigger>
        </TabsList>
        <TabsContent value="media" className="p-4">
          <div className="border rounded-md p-4">
            {uploadedImage ? (
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="max-w-full rounded-md border"
              />
            ) : (
              <p>No image uploaded yet.</p>
            )}
          </div>
        </TabsContent>
        <TabsContent value="code" className="p-4">
          <ScrollArea className="h-[calc(100vh-8rem)]">
            <div className="mb-4">
              <SyntaxHighlighter
                language="python"
                style={gruvboxLight}
                customStyle={{
                  padding: "1rem",
                  borderRadius: "0.375rem",
                  backgroundColor: "var(--muted)",
                }}
                wrapLongLines={true}
              >
                {uploadedFile || ""}
              </SyntaxHighlighter>
            </div>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="result" className="p-4">
          <div className="border rounded-md p-4">
            {uploadedResult ? (
              <img
                src={uploadedResult}
                alt="Uploaded"
                className="max-w-full rounded-md border"
              />
            ) : (
              <p>No image uploaded yet.</p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
}
