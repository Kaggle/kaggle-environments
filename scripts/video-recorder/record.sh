#!/bin/bash
# Example use: docker run --rm -v "/usr/local/google/home/dominoweir/git/video-recorder/recordings:/app/recordings" web-recorder /app/basic_test_hands.txt

# Graceful shutdown handler
SHUTDOWN_IN_PROGRESS=false
cleanup() {
    if [ "$SHUTDOWN_IN_PROGRESS" = true ]; then
        echo ""
        echo "[Shutdown] Already shutting down, please wait..."
        return
    fi
    SHUTDOWN_IN_PROGRESS=true

    echo ""
    echo "[Shutdown] Caught interrupt, saving recording..."

    # Kill node process if running
    if [ -n "$NODE_PID" ] && kill -0 $NODE_PID 2>/dev/null; then
        echo "[Shutdown] Stopping Playwright..."
        kill -INT $NODE_PID 2>/dev/null
        wait $NODE_PID 2>/dev/null
    fi

    # Gracefully stop FFmpeg to finalize the video
    if [ -n "$FFMPEG_PID" ] && kill -0 $FFMPEG_PID 2>/dev/null; then
        echo "[Shutdown] Finalizing video (this may take a moment)..."
        kill -INT $FFMPEG_PID 2>/dev/null
        wait $FFMPEG_PID 2>/dev/null
    fi

    # Stop Xvfb
    if [ -n "$XVFB_PID" ] && kill -0 $XVFB_PID 2>/dev/null; then
        kill $XVFB_PID 2>/dev/null
    fi

    echo "[Shutdown] Recording saved to $OUTPUT_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Parse arguments
TARGET_CONTENT="$1" # Could be an ID to be used in a URL or a file path
CUSTOM_FILENAME="$2"
RECORDING_LENGTH="$3"  # Optional: recording length in seconds (default: 3 hours)

# Export recording length for Node script
export RECORDING_LENGTH="${RECORDING_LENGTH:-$((3*60*60))}"

# Generate output filename (use custom name or timestamp)
if [ -n "$CUSTOM_FILENAME" ]; then
    OUTPUT_FILE="/app/recordings/${CUSTOM_FILENAME}.mp4"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # Extract hands file base name without .txt extension
    HANDS_BASENAME=$(basename "$TARGET_CONTENT" .txt)
    OUTPUT_FILE="/app/recordings/recording_${HANDS_BASENAME}_${TIMESTAMP}.mp4"
fi

echo "Output will be saved to: $OUTPUT_FILE"

# Ensure recordings directory exists
mkdir -p /app/recordings

# 1. Cleanup potential stale locks
rm -f /tmp/.X99-lock

# 2. Start Xvfb (Virtual Display)
Xvfb :99 -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24" &
XVFB_PID=$!
sleep 1

# 2.5. Hide cursor
unclutter -idle 0 -root &

# 3. Start PulseAudio (Virtual Audio)
pulseaudio -D --exit-idle-time=-1
pacmd load-module module-virtual-sink sink_name=vss
pacmd set-default-sink vss

# 4. Start FFmpeg (The Recorder)
# We capture video from :99 and audio from the vss monitor
# Comment out the loglevel line for more verbose output
ffmpeg -f x11grab \
    -video_size "${SCREEN_WIDTH}x${SCREEN_HEIGHT}" \
    -i :99 \
    -f pulse \
    -i vss.monitor \
    -y \
    -c:v libx264 \
    -preset fast \
    -pix_fmt yuv420p \
    -c:a aac \
    -b:a 128k \
    -hide_banner -loglevel error \
    "$OUTPUT_FILE" &
FFMPEG_PID=$!

# 5. Run the Playwright Script - change this out to another recording script if needed
node ./recording-scripts/record-werewolf.js "$TARGET_CONTENT" &
NODE_PID=$!
wait $NODE_PID

# 6. Shutdown sequence
# Playwright is already closed by record.js. Now we stop FFmpeg gracefully.
kill -INT $FFMPEG_PID
wait $FFMPEG_PID

# Stop Xvfb
kill $XVFB_PID

echo "Recording complete. File saved to $OUTPUT_FILE"
