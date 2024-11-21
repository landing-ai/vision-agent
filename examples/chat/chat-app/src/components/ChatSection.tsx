"use client";

import { useState, useEffect } from "react";
import { Send, Upload, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface ChatSectionProps {
  uploadedImage: string | null;
  onUploadedImage: (image: string) => void;
  uploadedFile: string | null;
  onUploadedFile: (file: string) => void;
  uploadedResult: string | null;
  onUploadedResult: (result: string) => void;
}

interface Message {
  role: "assistant" | "user" | "observation";
  content: string;
  media?: string[];
}

interface MessageBubbleProps {
  message: Message;
}

const CollapsibleMessage = ({ content }: { content: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="flex items-center justify-between">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm">
            {isOpen ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </CollapsibleTrigger>
        <span className="text-sm font-medium">Observation</span>
      </div>
      <CollapsibleContent>
        <pre className="pt-2 bg-gray-100 p-2 rounded-md overflow-x-auto">
          <code className="text-sm">{content}</code>
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
};

const checkContent = (content: string) => {
  const finalizePlanMatch = content.match(
    /<finalize_plan>(.*?)<\/finalize_plan>/s,
  );
  const finalCodeMatch = content.match(/<final_code>(.*?)<\/final_code>/s);
  return !(finalizePlanMatch || finalCodeMatch);
};

const formatAssistantContent = (content: string) => {
  const thinkingMatch = content.match(/<thinking>(.*?)<\/thinking>/s);
  const pythonMatch = content.match(/<execute_python>(.*?)<\/execute_python>/s);
  const finalPlanJsonMatch = content.match(/<json>(.*?)<\/json>/s);

  const finalPlanJson = JSON.parse(
    finalPlanJsonMatch ? finalPlanJsonMatch[1] : "{}",
  );
  if ("plan" in finalPlanJson && "instructions" in finalPlanJson) {
    return (
      <>
        <div>{finalPlanJson.plan}</div>
        <pre className="bg-gray-800 text-white p-1.5 rounded mt-2 overflow-x-auto text-xs">
          <code style={{ whiteSpace: "pre-wrap" }}>
            {Array.isArray(finalPlanJson.instructions)
              ? "-" + finalPlanJson.instructions.join("\n-")
              : finalPlanJson.instructions}
          </code>
        </pre>
      </>
    );
  }

  if (thinkingMatch || pythonMatch) {
    return (
      <>
        {thinkingMatch && <div>{thinkingMatch[1]}</div>}
        {pythonMatch && (
          <pre className="bg-gray-800 text-white p-1.5 rounded mt-2 overflow-x-auto text-xs">
            <code>{pythonMatch[1].trim()}</code>
          </pre>
        )}
      </>
    );
  }
  return <></>;
};

export function MessageBubble({ message }: MessageBubbleProps) {
  return (
    <div
      className={`mb-4 ${
        message.role === "user"
          ? "ml-auto bg-primary text-primary-foreground"
            : message.role === "assistant"
              ? "mr-auto bg-muted"
              : "mr-auto bg-secondary"
      } max-w-[80%] rounded-lg p-3`}
    >
      {message.role === "observation" ? (
        <CollapsibleMessage content={message.content} />
      ) : message.role === "assistant" ? (
        formatAssistantContent(message.content)
      ) : (
        message.content
      )}
    </div>
  );
}

export function ChatSection({
  uploadedImage,
  onUploadedImage,
  uploadedFile,
  onUploadedFile,
  uploadedResult,
  onUploadedResult,
}: ChatSectionProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const input = form.elements.namedItem("message") as HTMLInputElement;

    if (input.value.trim()) {
      const userMessage = { role: "user", content: input.value } as Message;
      if (uploadedImage) {
        userMessage.media = [uploadedImage];
      }

      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);

      try {
        const response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updatedMessages),
        });

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        const data = await response.json();
        console.log("Recieved response:", data);
      } catch (error) {
        console.error("Error:", error);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Sorry, there was an error processing your request.",
          },
        ]);
      }

      input.value = "";
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64String = event.target?.result as string;
        onUploadedImage(base64String);
      };
      reader.readAsDataURL(file);
    } else {
      console.error("Please upload an image file");
    }
  };

  const handleFinalCode = (message: Message) => {
    const finalCodeMatch = message.content.match(
      /<final_code>(.*?)<\/final_code>/s,
    );
    if (finalCodeMatch) {
      const finalCode = finalCodeMatch[1];
      onUploadedFile(finalCode);
      if (message.media && message.media.length > 0) {
        onUploadedResult(message.media[message.media.length - 1]);
      }
    }
  };

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = (event) => {
      console.log("Recieved event", event);
      const data = JSON.parse(event.data);
      setMessages((prev) => [...prev, data]);
      handleFinalCode(data);
    };
    return () => ws.close();
  }, []);

  return (
    <Card className="flex flex-col h-[800px]">
      <ScrollArea className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages
            .filter((message) => checkContent(message.content))
            .map((message, i) => (
              <MessageBubble key={i} message={message} />
            ))}
        </div>
      </ScrollArea>
      <div className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleFileUpload}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => document.getElementById("file-upload")?.click()}
          >
            <Upload className="h-4 w-4" />
          </Button>
          <input
            name="message"
            className="flex-1 px-3 py-2 rounded-md border"
            placeholder="Type your message..."
          />
          <Button type="submit" size="icon">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </Card>
  );
}
