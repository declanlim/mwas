#!/bin/bash

# Check if a command is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

# Run the provided command
"$@"

# Get the exit status of the command
EXIT_STATUS=$?

# Check if the command was killed by a signal
if [ $EXIT_STATUS -ge 128 ]; then
    SIGNAL=$(($EXIT_STATUS - 128))
    echo "Command was killed by signal $SIGNAL" >&2

    # Check for OOM kill
    if [ $SIGNAL -eq 9 ]; then
        echo "Checking for OOM kill in syslog..."
        if grep -i "killed process $$" /var/log/syslog; then
            echo "Command was killed by OOM Killer" >&2
        fi
    fi
else
    echo "Command exited with status $EXIT_STATUS" >&2
fi
