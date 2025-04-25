# VisionAgentV2 Example App

This is an example application to demonstrate how to run VisionAgentV2 locally.  
It only works with the **V2 version** of VisionAgent and is mainly used for debugging — expect to find bugs and issues.

---

## Quick Start

### 1. Clone and Set Up
Make sure you have Python, Node.js, and `make` installed.

To install all dependencies (Python and Node):

```bash
make setup
```

### 2. Run the App

To start both the backend and frontend together:

```bash
make run
```

---

## Human-in-the-loop Mode

If you want to run with human-in-the-loop support:

1. Open `app.py`
2. Set:

```python
DEBUG_HIL = True
```

Note: Not all visualizations are supported — only **object detection** and **segmentation**.

---

## Screenshot

![screenshot](https://github.com/landing-ai/vision-agent/blob/main/assets/screenshot.png?raw=true)

---

## Troubleshooting

### Port Already in Use (e.g. 8000)

If you get an error like:

```
[Errno 48] Address already in use
```

Run the following command to kill the process using port 8000:

```bash
kill -9 $(lsof -i TCP:8000 | grep LISTEN | awk '{print $$2}')
```

> **Note**: you can change `8000` to any other port number in `run.sh`
