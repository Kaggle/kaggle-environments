#!/bin/bash

# .75*PTS = 1.25 speed
ffmpeg -i /app/recordings/your_recording.mp4 \
-vf "fps=30000/1001,setpts=0.75*PTS" \ 
-an \
-crf 23 \
/app/recordings/SPED_UP_your_recording.mp4