This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the conversation agent by moving up one directory to the `chat/` directory. Within that directory, use `poetry install` on Python 3.11 to install the agent dependencies, then run

```bash
fastapi run
```

to initialize the agent.

Then, run the development server for this frontend UI:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Notes on the demo
Currently, no state is stored client-side or server-side, so once the browser page is refreshed, the entire conversation is lost.

To ensure the user is not required to wait for the agent to plan everything before previewing previously completed conversations, temporary buttons enabling conversations to be saved/loaded from the clipboard have been added. Scrolling to the bottom of the message history (if any) reveals these buttons.

> [!WARNING]
> Any media that is part of the conversation is included directly in the conversation data, making the data put in the clipboard very large.
> Be careful about potential lag from text editors which attempt to parse the entirety of a file at once.

## Learn More about Next.js

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!
