# Web Video Recorder

A headless video recorder that captures audio and video from any website using Playwright, FFmpeg, and Docker.

## Features

- **Audio + Video** — Captures both via PulseAudio virtual sink
- **Signal-based stop** — Waits for a `.replay-complete-signal` element to appear
- **Headless** — Runs in Docker with Xvfb virtual display
- **Fullscreen** — No browser UI, no cursor
- **Configurable output** — Timestamp-based filenames or custom names

## Quick Start

### Build

```bash
docker build -t web-recorder .
```

### Run

```bash
# Basic usage (timestamp filename, 3 hour max)
docker run --rm -v "$(pwd)/recordings:/app/recordings" web-recorder "https://example.com"

# Custom filename
docker run --rm -v "$(pwd)/recordings:/app/recordings" web-recorder "https://example.com" "my_recording"

# Custom filename + 60 second timeout
docker run --rm -v "$(pwd)/recordings:/app/recordings" web-recorder "https://example.com" "my_recording" 60

# Timestamp filename + custom timeout
docker run --rm -v "$(pwd)/recordings:/app/recordings" web-recorder "https://example.com" "" 60
```

### Arguments

| Position | Name | Description | Default |
|----------|------|-------------|---------|
| 1 | URL | Website URL to record | Required |
| 2 | Filename | Output filename (without .mp4) | `recording_YYYYMMDD_HHMMSS` |
| 3 | Length | Max recording length in seconds | `10800` (3 hours) |

### Output

Recordings are saved to the mounted `/app/recordings` directory as MP4 files with:
- **Video**: H.264, 1920x1080
- **Audio**: AAC, 128kbps

## How It Works

1. **Xvfb** creates a virtual display
2. **PulseAudio** creates a virtual audio sink
3. **FFmpeg** records the display and audio
4. **Playwright** opens Chrome in fullscreen and navigates to the URL
5. Recording stops when `.replay-complete-signal` element appears or timeout is reached

## Customization

### Stop Signal

Edit `record.js` to change the selector that triggers recording stop:

```javascript
await page.waitForSelector('.replay-complete-signal', { timeout: timeoutMs });
```

### Resolution

Edit `Dockerfile` to change the screen resolution:

```dockerfile
ENV SCREEN_WIDTH=1920
ENV SCREEN_HEIGHT=1080
```

Also update the window size in `record.js`:

```javascript
'--window-size=1920,1080',
```

## Requirements

- Docker

## License

MIT
