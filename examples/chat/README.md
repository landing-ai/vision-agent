# VisionAgentV2 Example App

This is an example application to demonstrate how to run VisionAgentV2 locally.  
It only works with the **V2 version** of VisionAgent and is mainly used for debugging — expect to find bugs and issues.

![screenshot](https://github.com/landing-ai/vision-agent/blob/main/assets/screenshot.png?raw=true)

## Prerequisites

- Python 3.7 or higher
- Node.js 14 or higher
- npm (comes with Node.js)

## Quick Start

### 1. Setup

#### On Windows (PowerShell)

Run the setup script:

```powershell
python setup.py
```

#### On Linux/macOS (with Make)

```bash
make setup
```

### 2. Run the App

#### On Windows (PowerShell)

```powershell
python run.py
```

#### On Linux/macOS (with Make)

```bash
make run
```

This will:
- Launch the FastAPI backend
- Start the React frontend
- Open your browser to the application
- Handle proper cleanup when you press Ctrl+C

## Human-in-the-loop Mode

To enable human-in-the-loop support:

1. Open `.env`
2. Set:
   ```bash
   DEBUG_HIL=true
   ```

**Note:** Currently, only **object detection** and **segmentation** visualizations are supported.

## Configuration

### Changing Ports

To modify the frontend or backend port:

1. Open `.env`
2. Change the `PORT_BACKEND` or `PORT_FRONTEND` variables:
   ```bash
   PORT_BACKEND = 8000  # Change to your preferred port
   PORT_FRONTEND = 3000  # Change to your preferred port
   ```

## Troubleshooting

- **Port conflicts**: The run script will attempt to free ports if they're already in use, but if not, either use another port or kill the process that is currently running on the conflicting port (make sure you know what is running on this port before killing it)
- **Services not starting**: Verify you have the prerequisites installed and ran the setup
- **Browser doesn't open**: Manually navigate to http://localhost:3000 (or whatever your frontend port is)
- **Constant string of messages saying connection rejected/closed**: Check that you do not have multiple tabs open to http://localhost:3000 (or whatever your frontend port is)

## Support

For issues and questions, please file an issue on the GitHub repository, or come ask questions in our Discord: 

[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
