#!/bin/bash
ffmpeg -framerate 25 -i ./data/pendulum/gcl_itr%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./data/pendulum/gcl_output.mp4