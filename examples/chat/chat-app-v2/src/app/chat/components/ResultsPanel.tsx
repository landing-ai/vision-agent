import { Button } from "@/components/ui/Button";
import * as Tabs from "@radix-ui/react-tabs";
import Image from "next/image";

{/* TODO: Add a component which renders code and result images */}
export interface ResultsPanelProps {
  code: string | null;
  images: string[] | null;
  backButton: JSX.Element;
}

export default function ResultsPanel({ code, images, backButton }: ResultsPanelProps): JSX.Element {
  // const [activeTab, setActiveTab]

  return (<div className="relative flex flex-col gap-2 h-full overflow-hidden">
    {backButton}
    <Tabs.Root defaultValue="code" className="shrink grow flex flex-col gap-2">
      <Tabs.List className="flex gap-2">
        <Tabs.Trigger value="code" className="py-1 px-2 rounded bg-gray-700 data-[state=active]:bg-green-600">
          Code
        </Tabs.Trigger>
        <Tabs.Trigger value="result" className="py-1 px-2 rounded bg-gray-700 data-[state=active]:bg-green-600">
          Result
        </Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="code" className="shrink h-full overflow-hidden">
        <pre className="overflow-y-auto h-full">
          <code>
            {code}
          </code>
        </pre>
      </Tabs.Content>
      <Tabs.Content value="result" className="shrink overflow-hidden h-full">
        {images?.map((image, index) =>
          <Image
            src={image}
            alt={`Processed image ${index}`}
            key={index}
            fill
            className={"object-contain"} />)}
      </Tabs.Content>
    </Tabs.Root>
  </div>)
}
