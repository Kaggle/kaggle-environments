FROM mcr.microsoft.com/playwright:v1.40.0-focal

USER root
RUN apt-get update && apt-get install -y xvfb pulseaudio ffmpeg procps unclutter

# Forces Playwright to use the browsers already in the image
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV DISPLAY=:99
ENV SCREEN_WIDTH=1920
ENV SCREEN_HEIGHT=1080

WORKDIR /app
COPY package.json .
RUN npm install

# Double-check browsers are there for the version in package.json
RUN npx playwright install chromium

COPY record.sh .
COPY speed_up_video.sh .
COPY record.js .
COPY login.js .
COPY recording-scripts/ ./recording-scripts/

# Mount point for hands files passed at runtime
RUN mkdir -p /app/hands
RUN chmod +x record.sh speed_up_video.sh

ENTRYPOINT ["./record.sh"]