# Example App
This is an example applicaiton to demonstrate how to run VisionAgent locally.


# Quick Start
To get started install the fastapi library:
```bash
pip install "fastapi[standard]"
```

Then cd into `chat-app` and run `make` to install the node dependencies.

To start the server run:
```bash
fastapi dev app.py
```

And to start the front end, in the chat-app folder run:
```bash
npm run dev
```

Then go to `http://localhost:3000` in your browser to see the app running.

![screenshot](https://github.com/landing-ai/vision-agent/blob/main/assets/screenshot.png?raw=true)
