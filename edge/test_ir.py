import time
import sys

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Error: RPi.GPIO is not installed or you are not running this on a Raspberry Pi.")
    sys.exit(1)

# BCM Pins from your config
IR_FRONT_PIN = 17
IR_BACK_PIN = 27
IR_LEFT_PIN = 22
IR_RIGHT_PIN = 23

SENSORS = {
    "Front": IR_FRONT_PIN,
    "Back": IR_BACK_PIN,
    "Left": IR_LEFT_PIN,
    "Right": IR_RIGHT_PIN,
}

def main():
    print("========================================")
    print("      IR Sensor Standalone Test")
    print("========================================")
    print("Initializing IR sensors...")
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        for name, pin in SENSORS.items():
            # Pull-up resistor ensures it reads HIGH (1) when nothing is triggering it
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(f"Configured {name} sensor on GPIO {pin} (BCM)")
            
        print("\nStarting read loop. Press Ctrl+C to stop.")
        print("-" * 50)
        
        while True:
            output = []
            for name, pin in SENSORS.items():
                val = GPIO.input(pin)
                # Active low: 0 means obstacle detected, 1 means clear
                if val == 0:
                    status = "[OBSTACLE]"
                else:
                    status = "  Clear   "
                output.append(f"{name}: {status}")
                
            # Print on the same line to make it easy to read
            print(" | ".join(output), end="\r", flush=True)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up. Goodbye!")

if __name__ == "__main__":
    main()
