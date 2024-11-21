'use client';

import { createRef, useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronUp, Send, Upload } from "lucide-react";
import Textarea from "react-textarea-autosize";

import { Message, ChatParticipant, agentResponseToJSX, assistantMessageToAgentResponse } from "@/lib/agent";
import Image from "next/image";
import { IconArrowDown, IconClose } from "@/components/Icons";
import { cn } from "@/lib/utils";
import ResultsPanel from "./ResultsPanel";
import { Button } from "@/components/ui/Button";

export const CollapsibleMessage = ({ content, title }: { content: string, title: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="grid grid-rows-[20px_0fr] data-[is-open=true]:gap-2 data-[is-open=true]:grid-rows-[20px_1fr] group" data-is-open={isOpen ? "true" : undefined}>
      <button onClick={() => setIsOpen(!isOpen)} className="flex gap-2 items-center">
        {isOpen ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
        <p>{title}</p>
      </button>

      <pre className="hidden opacity-0 group-data-[is-open=true]:block group-data-[is-open=true]:opacity-100 pt-2 bg-gray-800 p-2 rounded-md overflow-x-auto text-sm">
        <code>{content}</code>
      </pre>
    </div>
  );
};

const MessageGroup = ({ message }: { message: Message }) => {
  return (<>
    {message.media && message.media.map((mediaURL, index) => (
      <div className={`rounded-xl overflow-hidden max-w-xs relative ${message.role === ChatParticipant.User ? "rounded-br-none self-end" : "rounded-bl-none self-start"}`} key={index}>
        <Image
          src={mediaURL}
          width={100}
          height={100}
          alt={`user uploaded image ${index}`}
          sizes="100vw"
          style={{
            width: "100%",
            height: "auto"
          }} />
      </div>
    ))}
    <MessageBubble message={message} />
  </>);
}

export const MessageBubble = ({ message }: { message: Message }) => {
  let participantSpecificStyling = "rounded-bl-none self-start";
  let BubbleContent: JSX.Element | null = null;
  switch (message.role) {
    case ChatParticipant.User:
      participantSpecificStyling = "rounded-br-none self-end bg-gray-700";
      BubbleContent = (<span>{message.content}</span>);
      break;
    case ChatParticipant.Planner:
    case ChatParticipant.Coder:
    case ChatParticipant.Conversation:
    case ChatParticipant.Assistant:
      participantSpecificStyling += " bg-muted";
      BubbleContent = agentResponseToJSX(assistantMessageToAgentResponse(message));
      break;
    case ChatParticipant.Observation:
      participantSpecificStyling += " bg-secondary";
      BubbleContent = (<CollapsibleMessage content={message.content} title={"[Observation]"} />);
      break;
    default:
      BubbleContent = (<span>{message.content}</span>);
      break;
  }

  if (!BubbleContent) {
    return <></>;
  }


  return (
    <div className={`flex flex-col gap-2 p-2 rounded-xl ${participantSpecificStyling} mb-4`}>
      {![ChatParticipant.User, ChatParticipant.Observation].includes(message.role) && <p className="uppercase font-bold text-sm">{`[${message.role}]`}</p>}
      {BubbleContent}
    </div>
  );
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);

  const [sidePanelOpen, setSidePanelOpen] = useState<boolean>(false);

  const [uploadedCode, setUploadedCode] = useState<string | null>(null);
  const [uploadedResultImage, setUploadedResultImage] = useState<string | null>(null);

  const [userImage, setUserImage] = useState<string | null>(null);
  const [clearInputKey, setClearInputKey] = useState<number>(Date.now());
  const ForceClearInput = () => setClearInputKey(Date.now());

  const handleFileUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = e.target.files;
      console.log(selectedFiles);
      if (!selectedFiles || selectedFiles.length < 1) return;
      const file = selectedFiles[0];
      if (!file || !file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        return;
      }

      const reader = new FileReader();
      reader.onload = (event)=> {
        const base64String = event.target?.result as string;
        setUserImage(base64String);
      };
      reader.readAsDataURL(file);
    },
    []
  );
  const removeUserImage = () => {
    ForceClearInput();
    setUserImage(null);
  }

  const handleUserMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const prompt = form.elements.namedItem("message") as HTMLTextAreaElement;

    if (!prompt.value.trim()) return;

    const userMessage = new Message(ChatParticipant.User, prompt.value);
    if (userImage) {
      userMessage.media = [userImage];
    }

    const allMessages = [...messages, userMessage];
    setMessages(allMessages);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(allMessages),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("Received response:", data);
    } catch (error) {
      console.error("[MESSAGE POST ERR]", error);
      setTimeout(() => setMessages((prev) => [
        ...prev,
        new Message(ChatParticipant.Assistant, "<response>Sorry, there was an error processing your request.</response>")
      ]), 0);
    }

    // Clear input
    prompt.value = "";
    removeUserImage();
  }

  const tryFindingFinalCode = (message: Message) => {
    const finalCodeMatch = message.content.match(
      /<final_code>(.*?)<\/final_code>/s,
    );
    if (!finalCodeMatch) return;
    const finalCode = finalCodeMatch[1];
    setUploadedCode(finalCode);
    if (message.media && message.media.length > 0) {
      setUploadedResultImage(message.media[message.media.length - 1]);
    }
    // TODO: Re-enable the user's message composer and open the side panel
    setSidePanelOpen(true);
  }

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = (event) => {
      console.log("[Websocket] Received event", event);

      const data = JSON.parse(event.data) as Message;
      setMessages(messages => [...messages, data]);
      tryFindingFinalCode(data);
    };
    return () => ws.close();
  }, []);

  // DEBUG


  useEffect(() => {
    if (messageFeedRef.current) {
      messageFeedRef.current.scrollTop = messageFeedRef.current.scrollHeight;
    }
  }, [messages]);

  const messageFeedRef = createRef<HTMLDivElement>();
  const fileInputRef = createRef<HTMLInputElement>();

  return (
    (<>
      <div
        className={`size-full grid grid-cols-[1fr_0fr] md:px-20 lg:px-36 data-[side-panel-open=true]:px-0 data-[side-panel-open=true]:grid-cols-[2fr_3fr] transition-all duration-300 group`}
        data-side-panel-open={sidePanelOpen ? "true" : undefined}>
        <div className="relative overflow-hidden">
          <div
            ref={messageFeedRef}
            className="flex flex-col items-center justify-between gap-4 overflow-y-auto hide-scrollbar px-8 h-full">
            <div className="flex flex-col items-center gap-2 w-full">
              {messages
                .map((message, index) => <MessageGroup message={message} key={index} />)}

              <Button variant="ghost" className="bg-gray-800 rounded-full p-2 w-min" onClick={() => setSidePanelOpen(true)}>
                TEMP: Open Side Panel
              </Button>
            </div>


            <div className="sticky bottom-4 p-2 mt-4 rounded-lg bg-gray-900 border w-full md:w-[42rem] z-50">
              <form className="flex h-full gap-4 items-start justify-between" onSubmit={handleUserMessage}>
                <div className="flex-1">
                  {userImage && <div className="mb-2">
                    <div className="size-16 relative">
                      <Image
                        src={userImage}
                        className="object-cover rounded-md"
                        alt={"User-submitted image"}
                        fill />
                      <button className="absolute top-[-0.375rem] right-[-0.375rem] rounded-full bg-slate-400 hover:bg-slate-300 p-[0.1rem] group" onClick={(e) => {e.preventDefault(); e.stopPropagation(); removeUserImage()}}>
                        <IconClose className={cn(`size-3 text-red-700 group-hover:text-red-500 pointer-events-none`)} />
                      </button>
                    </div>
                  </div>}
                  <div className="flex items-center">
                    <Textarea
                      rows={1}
                      name="message"
                      placeholder="Message Planner v2..."
                      required={true}
                      className="flex-1 w-full bg-transparent rounded p-1 border-0 outline-0 resize-none" />
                  </div>
                </div>

                <input
                  type="file"
                  key={clearInputKey}
                  ref={fileInputRef}
                  accept="image/png,image/jpeg,image/jpg,image/webp,image/heic,image/avif"
                  id="user-file-upload"
                  className="hidden"
                  onChange={handleFileUpload} />
                <div className="flex gap-2">
                  <button className="rounded border border-gray-600 bg-gray-700 p-2" onClick={(e) => {e.preventDefault(); document.getElementById("user-file-upload")?.click()}}>
                    <Upload className="size-4" />
                  </button>
                  <button className="rounded border border-green-600 bg-green-700 p-2" type="submit">
                    <Send className="size-4" />
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
      {/* results panel */}
      <div className={`absolute px-4 top-0 left-[40vw] w-[60vw] h-full hidden lg:flex flex-col overflow-hidden transition-transform duration-300 ${sidePanelOpen ? "translate-x-0" : "translate-x-full"}`}>
        <ResultsPanel code={uploadedCode} images={uploadedResultImage ? [uploadedResultImage] : null}
          backButton={(<Button
            size="sm"
            variant="ghost"
            className="flex w-min p-2 items-center text-sm"
            onClick={() => {
              setSidePanelOpen(false)
            }}
          >
            <IconArrowDown className="mr-1 size-4 rotate-90" />
            Back to conversation
          </Button>)}
        />
      </div>
    </>)
  );
}
