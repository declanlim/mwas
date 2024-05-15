#!/bin/bash

# Run the process
your_command_here &

# Get the PID of the process
PID=$!

# Wait for the process to finish
wait $PID

# Get the exit status of the process
EXIT_STATUS=$?

# Check if the process was killed by a signal
if [ $EXIT_STATUS -ge 128 ]; then
    SIGNAL=$(($EXIT_STATUS - 128))
    echo "Process was killed by signal $SIGNAL" >&2

    # Check for OOM kill
    if [ $SIGNAL -eq 9 ]; then
        echo "Checking for OOM kill in syslog..."
        if grep -i "killed process $PID" /var/log/syslog; then
            echo "Process was killed by OOM Killer" >&2
        fi
    fi
else
    echo "Process exited with status $EXIT_STATUS" >&2
fi
