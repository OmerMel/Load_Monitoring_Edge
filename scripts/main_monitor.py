import time
import subprocess
import sys
import os
from datetime import datetime

# ------------- Configuration ------------- #
INTERVAL_SECONDS = 20  # 2 minutes by default
SCRIPT_PATH = os.path.join(os.path.dirname(
    __file__), "live_stream_processor.py")
PYTHON_EXECUTABLE = sys.executable

# ---------------------------------------------------------------------------------------------------------------#
# Function to display a countdown timer on the same line in the terminal
# Args: Number of seconds to countdown


def run_countdown(seconds):
    try:
        # Initial newline to separate from previous output
        print()
        for remaining in range(seconds, 0, -1):
            sys.stdout.write(f"\rNext execution in: {remaining}s...   ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\rExecuting now! \n")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nCountdown interrupted.")
        raise

# ---------------------------------------------------------------------------------------------------------------#
# Main function to orchestrate the train carriage load monitoring system


def main():
    # Print the starting message
    print(f"Starting Main Monitor.")
    print(f"Target Script: {SCRIPT_PATH}")
    print(f"Interval: {INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    while True:
        try:
            # 1. Log start time
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{start_time}] Starting processing cycle...")

            # 2. Execute the live stream processor in single-shot mode
            # We pass --single-shot so it runs once and exits
            # We also pass any other necessary default args if needed,
            # but live_stream_processor.py has defaults.
            command = [PYTHON_EXECUTABLE, SCRIPT_PATH, "--single-shot"]

            # Run the subprocess
            result = subprocess.run(command, text=True)

            # 3. Log completion
            if result.returncode == 0:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Cycle completed successfully.")
            else:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Cycle finished with error code: {result.returncode}")

            # 4. Countdown to next execution
            run_countdown(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n\nStopping Main Monitor. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)


if __name__ == "__main__":
    main()
