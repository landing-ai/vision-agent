# Example App
This is an example applicaiton to demonstrate how to run VisionAgentV2 locally. This
only works with the V2 version of VisionAgent and is mainly used for debugging, expect
to find bugs and issues.


# Quick Start
To get started install the fastapi library:
```bash
pip install "fastapi[standard]"
```

Then cd into `chat-app` and run `make` to install the node dependencies.

To start the server run:
```bash
./run.sh
```

And to start the front end, in the chat-app folder run:
```bash
npm run dev
```

Then go to `http://localhost:3000` in your browser to see the app running.

![screenshot](https://github.com/landing-ai/vision-agent/blob/main/assets/screenshot.png?raw=true)

# Human-in-the-loop
If you want to run with human-in-the-loop, set `DEBUG_HIL = True` inside `app.py`. Note
not all visualizations are supported, only object detection and segmentation for images.
