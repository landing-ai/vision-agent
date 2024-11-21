import * as Tabs from "@radix-ui/react-tabs";
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import Image from "next/image";
import { useState } from "react";

{/* TODO: Add a component which renders code and result images */}
export interface ResultsPanelProps {
  code: string | null;
  images: string[] | null;
  backButton: JSX.Element;
}

export default function ResultsPanel({ code, images, backButton }: ResultsPanelProps): JSX.Element {
  const [activeTab, setActiveTab] = useState<string>("code");

  return (<div className="relative flex flex-col gap-2 h-full overflow-hidden">
    {backButton}
    <div className="flex flex-col gap-2 h-full overflow-hidden">
      <ToggleGroup.Root
        type="single"
        defaultValue="code"
        aria-label="Execution result tab"
        className="flex gap-2"
        onValueChange={(value) => {if (value) setActiveTab(value)}}>
        <ToggleGroup.Item value="code" className="py-1 px-2 rounded bg-gray-700 data-[state=on]:bg-green-600">
          Code
        </ToggleGroup.Item>
        <ToggleGroup.Item value="result" className="py-1 px-2 rounded bg-gray-700 data-[state=on]:bg-green-600">
          Result
        </ToggleGroup.Item>
      </ToggleGroup.Root>
      <Tabs.Root value={activeTab} className="flex-1 overflow-hidden">
        {/* <Tabs.List className="flex gap-2">
          <Tabs.Trigger value="code" className="py-1 px-2 rounded bg-gray-700 data-[state=active]:bg-green-600">
            Code
          </Tabs.Trigger>
          <Tabs.Trigger value="result" className="py-1 px-2 rounded bg-gray-700 data-[state=active]:bg-green-600">
            Result
          </Tabs.Trigger>
        </Tabs.List> */}
        <Tabs.Content value="code" className="shrink h-full overflow-y-auto">
          <pre className="rounded-sm bg-gray-800 p-2">
            <code>
              {code}
            </code>
          </pre>
        </Tabs.Content>
        <Tabs.Content value="result" className="relative shrink h-full overflow-y-auto">
          {images?.map((image, index) =>
            <Image
              src={image}
              alt={`Processed image ${index}`}
              key={index}
              fill
              className={"object-contain"} />)}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  </div>)
}
