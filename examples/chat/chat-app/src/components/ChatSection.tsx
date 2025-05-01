"use client";

import { GroupedVisualizer } from "@/components/GroupedVisualizer";
import { Polygon } from "@/components/PolygonDrawer";
import { useState, useEffect, useRef } from "react";
import { Send, Upload, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import Prism from "prismjs";
import "prismjs/themes/prism-tomorrow.css";
import "prismjs/components/prism-python.min.js"
import ReactMarkdown from "react-markdown";
import remarkGfm from 'remark-gfm' // For tables, strikethrough
import rehypeHighlight from 'rehype-highlight' // For code syntax highlighting



interface ChatSectionProps {
  uploadedMedia: string | null;
  onUploadedMedia: (image: string) => void;
  uploadedFile: string | null;
  onUploadedFile: (file: string) => void;
  uploadedResult: string | null;
  onUploadedResult: (result: string) => void;
  polygons: Polygon[];
}

interface Message {
  role:
    | "assistant"
    | "conversation"
    | "interaction"
    | "interaction_response"
    | "coder"
    | "planner"
    | "planner_update"
    | "user"
    | "observation";
  content: string;
  media?: string[];
}

interface MessageBubbleProps {
  message: Message;
  onSubmit?: (functionName: string, boxThreshold: number) => void;
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
        <pre className="pt-2 bg-gray-100 p-2 rounded-md overflow-x-auto max-w-full whitespace-pre-wrap">
          <code className="text-sm">{content}</code>
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
};

const checkContent = (role: string, content: string) => {
  const finalizePlanMatch = content.match(
    /<finalize_plan>(.*?)<\/finalize_plan>/s,
  );
  const finalCodeMatch = content.match(/<final_code>(.*?)<\/final_code>/s);
  return (
    !(finalizePlanMatch || finalCodeMatch) &&
    !role.includes("update") &&
    !role.includes("response")
  );
};

const formatAssistantContent = (
  role: string,
  content: string,
  onSubmit: (functionName: string, boxThreshold: number) => void,
) => {
  useEffect(() => {
    Prism.highlightAll();
  }, [content]);
  const responseMatch = content.match(/<response>(.*?)<\/response>/s);
  const thinkingMatch = content.match(/<thinking>(.*?)<\/thinking>/s);
  const pythonMatch = content.match(/<execute_python>(.*?)<\/execute_python>/s);
  const finalPlanJsonMatch = content.match(/<json>(.*?)<\/json>/s);
  const interactionMatch = content.match(/<interaction>(.*?)<\/interaction>/s);
  let interactionJson = JSON.parse(
    interactionMatch ? interactionMatch[1] : "[]",
  );
  interactionJson = interactionJson.filter(
    (elt: { type: string }) => elt.type === "tool_func_call",
  );

  const finalPlanJson = JSON.parse(
    finalPlanJsonMatch ? finalPlanJsonMatch[1] : "{}",
  );
  if ("plan" in finalPlanJson && "instructions" in finalPlanJson) {
    return (
      <>
        <div>
          <strong className="text-gray-700">[{role.toUpperCase()}]</strong>{" "}
          {finalPlanJson.plan}
        </div>
        <pre className="bg-gray-800 text-white p-1.5 rounded mt-2 overflow-x-auto text-xs]">
          <code style={{ whiteSpace: "pre-wrap" }}>
            {Array.isArray(finalPlanJson.instructions)
              ? "-" + finalPlanJson.instructions.join("\n-")
              : finalPlanJson.instructions}
          </code>
        </pre>
      </>
    );
  }

  if (interactionMatch) {
    return (
      <GroupedVisualizer
        detectionResults={interactionJson}
        onSubmit={onSubmit}
      />
    );
  }

  if (responseMatch || thinkingMatch || pythonMatch) {
    return (
      <>
        {thinkingMatch && (
          <div>
            <strong className="text-gray-700">[{role.toUpperCase()}]</strong>{" "}
            {thinkingMatch[1]}
          </div>
        )}
        {responseMatch && (
          <div>
            <strong className="text-gray-700">[{role.toUpperCase()}]</strong>{" "}
            <div className="prose prose-sm max-w-none markdown">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]} 
                rehypePlugins={[rehypeHighlight]}
              >
                {responseMatch[1]}
              </ReactMarkdown>
            </div>
          </div>
        )}
        {pythonMatch && (
          <pre className="bg-gray-800 text-white p-1.5 rounded mt-2 overflow-x-auto text-xs max-w-full whitespace-pre-wrap" style={{ fontSize: '12px' }}>
            <code
              className="language-python"
              dangerouslySetInnerHTML={{
                __html: Prism.highlight(pythonMatch[1].trim(), Prism.languages.python, "python"),
              }}
            />
          </pre>
        )}
      </>
    );
  }
  return <></>;
};

function MessageBubble({ message, onSubmit }: MessageBubbleProps) {
  const handleSubmit = onSubmit || ((functionName: string, boxThreshold: number) => {
    console.log("No onSubmit handler provided", functionName, boxThreshold);
  });
  return (
      <div
        className={`mb-4 max-w-[80%] p-3 rounded-lg shadow-sm ${
          message.role === "user" || message.role === "interaction_response"
            ? "ml-auto bg-blue-600 text-white"
            : message.role === "assistant"
            ? "mr-auto bg-gray-100 text-gray-900"
            : "mr-auto bg-gray-200 text-gray-800"
        }`}
      >
      {message.role !== "user" && (
        <div className="mb-1 text-xs text-gray-500 uppercase font-medium">
          {message.role}
        </div>
      )}
      {message.role === "observation" ? (
        <CollapsibleMessage content={message.content} />
      ) : message.role === "assistant" ||
        message.role === "conversation" ||
        message.role === "planner" ||
        message.role === "interaction" ||
        message.role === "coder" ? (
        formatAssistantContent(message.role, message.content, handleSubmit)
      ) : (
        message.content
      )}
    </div>
  );
}

export function ChatSection({
  uploadedMedia,
  onUploadedMedia,
  uploadedFile,
  onUploadedFile,
  uploadedResult,
  onUploadedResult,
  polygons,
}: ChatSectionProps) {
  const port_backend = process.env.PORT_BACKEND;
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const handleCancel = async () => {
    try {
      const response = await fetch(`http://localhost:${port_backend}/cancel`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        }
      });
  
      if (!response.ok) {
        console.error("Failed to cancel processing:", await response.text());
      } else {
        console.log("Processing canceled successfully");
      }
      
      setMessages([]);
      setIsTyping(false);
      if (uploadedMedia) {
        onUploadedMedia("");
      }
    } catch (error) {
      console.error("Error canceling processing:", error);
    }
  };
  const TypingIndicator = () => {
    return (
      <div className="flex items-center space-x-1 p-2 rounded-lg bg-gray-100 w-12">
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "0ms" }}></div>
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "300ms" }}></div>
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "600ms" }}></div>
      </div>
    );
  };
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);    

  const sendMessages = async (messages: Message[]) => {
    try {
      const lastMessage = {...messages[messages.length - 1]};
      if (polygons.length > 0 && lastMessage.role === "user") {
        const polygonStrings = polygons.map(polygon => 
          `${polygon.name}: [${polygon.points.map(p => `(${p.x}, ${p.y})`).join(', ')}]`
        );
        lastMessage.content += "\nPolygons: " + polygonStrings.join('; ');
      }
      
      console.log("Sending message:", lastMessage);
      messages[messages.length - 1] = lastMessage;
      setIsTyping(true);
      const response = await fetch(`http://localhost:${port_backend}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(messages),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("Recieved response:", data);
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Request canceled or failed." },
      ]);
    }
  };

  const interactionCallback = async (
    functionName: string,
    boxThreshold: number,
  ) => {
    const message = {
      role: "interaction_response",
      content: JSON.stringify({
        function_name: functionName,
        box_threshold: boxThreshold,
      }),
    } as Message;
    const updatedMessages = [...messages, message];
    setMessages(updatedMessages);
    sendMessages(updatedMessages);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const input = form.elements.namedItem("message") as HTMLInputElement;

    if (input.value.trim()) {
      let userMessage: Message;
      if (
        messages.length > 0 &&
        messages[messages.length - 1].role === "interaction"
      ) {
        // this is handled by the interaction callback
        return;
      } else {
        userMessage = { role: "user", content: input.value } as Message;
        if (uploadedMedia) {
          userMessage.media = [uploadedMedia];
        }
      }

      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);
      sendMessages(updatedMessages);

      input.value = "";
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];

    if (
      file &&
      (file.type.startsWith("image/") || file.type.startsWith("video/"))
    ) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64String = event.target?.result as string;
        onUploadedMedia(base64String);
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
    let ws: WebSocket;
    let reconnectInterval: NodeJS.Timeout;
  
    const connect = () => {
      ws = new WebSocket(`ws://localhost:${port_backend}/ws`);
  
      ws.onopen = () => {
        console.log("✅ WebSocket connected");
      };
  
      ws.onmessage = (event) => {
        console.log("Received event", event);
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, data]);
        handleFinalCode(data);
        setIsTyping(false);
      };
  
      ws.onclose = () => {
        console.warn("WebSocket closed. Reconnecting...");
        reconnectInterval = setTimeout(connect, 1000); // Retry after 1s
      };
  
      ws.onerror = (err) => {
        console.error("WebSocket error (backend likely shut down)", err);
        ws.close(); // Triggers onclose -> reconnect
      };
    };
  
    connect();
  
    return () => {
      if (ws) ws.close();
      if (reconnectInterval) clearTimeout(reconnectInterval);
    };
  }, []);
  

  return (
    <Card className="flex flex-col h-[800px]">
      <ScrollArea className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages
            .filter((message) => checkContent(message.role, message.content))
            .map((message, i) => (
              <MessageBubble
                key={i}
                message={message}
                onSubmit={interactionCallback}
              />
            ))}
          {isTyping && (
            <div className="mr-auto">
              <TypingIndicator />
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex gap-2 items-center bg-white border rounded-lg p-2 shadow-md">
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
          <Button type="button" variant="secondary" size="icon" onClick={handleCancel}>
            ✖
          </Button>

        </form>
      </div>
    </Card>
  );
}