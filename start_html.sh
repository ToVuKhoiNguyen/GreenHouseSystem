#!/bin/bash

SESSION_NAME="tailscale_web"
# Set your directory path here. Use an absolute path if the script isn't run from the parent directory.
TARGET_DIR="GreenHouseSystem" 
cd $TARGET_DIR
# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
  echo "Tmux session '$SESSION_NAME' already exists. Attaching to it..."
  tmux attach-session -t $SESSION_NAME
  exit 0
fi

# Create a new detached tmux session
tmux new-session -d -s $SESSION_NAME

# Pane 0 (Left): Start the Python HTTP server targeting the specific directory
tmux send-keys -t $SESSION_NAME:0 "sudo python3 -m http.server 8443" C-m

# Split the window horizontally
tmux split-window -h -t $SESSION_NAME:0

# Pane 1 (Right): Start the Tailscale funnel
tmux send-keys -t $SESSION_NAME:0.1 "sudo tailscale funnel 8443" C-m

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME
