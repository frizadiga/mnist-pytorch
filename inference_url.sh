#!/bin/bash
# Download image from URL and run inference
# Usage: ./inference_url.sh <image_url>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <image_url>"
    echo "Example: $0 https://example.com/digit.png"
    exit 1
fi

URL="$1"
TEMP_DIR="temp"
TEMP_FILE="$TEMP_DIR/downloaded_image.png"

# Create temp directory
mkdir -p "$TEMP_DIR"

echo "Downloading: $URL"

# Download image
curl -L -s -o "$TEMP_FILE" "$URL" || {
    echo "Failed to download image"
    exit 1
}

echo "Running inference..."

# Run inference with temp file path as argument
python3 inference.py "$TEMP_FILE"

# Cleanup
rm -f "$TEMP_FILE"

echo "Done!"
