"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface VersionSelectorProps {
  onVersionSelect: (version: 'v2' | 'v3') => void;
}

export function VersionSelector({ onVersionSelect }: VersionSelectorProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="p-8 max-w-md w-full mx-4 bg-white">
        <div className="text-center space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">
            Choose Version
          </h2>
          <p className="text-gray-600">
            Select which version of the chat API you would like to use:
          </p>
          <div className="space-y-3">
            <Button
              onClick={() => onVersionSelect('v3')}
              className="w-full py-3 text-lg border border-gray-300 bg-white text-gray-900 hover:bg-gray-50"
              variant="outline"
            >
              VisionAgentV3
            </Button>
            <Button
              onClick={() => onVersionSelect('v2')}
              className="w-full py-3 text-lg border border-gray-300 bg-white text-gray-900 hover:bg-gray-50"
              variant="default"
            >
              VisionAgentV2
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
