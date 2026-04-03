import time
import board
import busio
import adafruit_mlx90640

# ANSI color codes for our terminal "heatmap"
COLORS = ['\033[44m  \033[0m',  # Deep Blue (Coldest)
          '\033[46m  \033[0m',  # Cyan
          '\033[42m  \033[0m',  # Green
          '\033[43m  \033[0m',  # Yellow
          '\033[41m  \033[0m',  # Red (Hottest)
          '\033[45m  \033[0m']  # Magenta (Burning)

# Setup I2C
print("Initializing MLX90640 Thermal Camera...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

frame = [0.0] * 768

while True:
    try:
        mlx.getFrame(frame)
        
        # Calculate dynamic range to scale the colors
        min_temp = min(frame)
        max_temp = max(frame)
        range_temp = max_temp - min_temp if max_temp != min_temp else 1
        
        # Move terminal cursor to top-left to draw a smooth "video" update
        print('\033[2J\033[H', end="")
        print(f"--- LIVE THERMAL STREAM: Max Temp: {max_temp:.1f}°C ---")

        # Draw the 32x24 grid row by row
        for h in range(24):
            line = ""
            for w in range(32):
                temp = frame[h * 32 + w]
                # Normalize temperature between 0.0 and 1.0 based on current scene
                norm = (temp - min_temp) / range_temp
                
                # Assign to one of the 6 colors
                color_idx = int(norm * (len(COLORS) - 1))
                line += COLORS[color_idx]
            print(line)
            
    except ValueError:
        pass # Ignore missed internal subpages
    except Exception as e:
        print(f"Error: {e}")