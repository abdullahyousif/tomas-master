#!/bin/bash

# Create directory structure
mkdir -p config models logs data

# Copy settings.json to config directory if it exists
if [ -f "settings.json" ]; then
  cp settings.json config/
  echo "Copied settings.json to config directory"
else
  echo "Warning: settings.json not found"
fi

# Create README.md in models directory
cat > models/README.md << 'EOL'
# YOLOv11 Model Directory

Place your YOLOv11 model file (`my_model.pt`) in this directory.

Make sure the model is trained to detect the following game elements:

- attack_aiming_icon
- attack_symbol
- raid_hole_icon
- raid_symbol
- raid_x_icon
EOL

echo "Directory structure created!"
echo "Please place your YOLOv11 model file (my_model.pt) in the models directory"
echo "Run the bot with: python main.py"
