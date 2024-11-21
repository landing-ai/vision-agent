import Image from "next/image";

import Chat from "./chat/components/ChatWrapper";

export default function Home() {
  return (
    <div className="h-screen py-8 gap-4 font-[family-name:var(--font-geist-sans)]">
      <main className="relative flex flex-col size-full gap-8 items-center sm:items-start overflow-x-hidden">
        <Chat />
      </main>
    </div>
  );
}
