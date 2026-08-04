# Pi Edge Passenger Load Monitor

Pi Edge is the Raspberry Pi application responsible for collecting passenger-load data inside a train carriage. It performs person detection locally on camera images, tracks entry and exit events with a pair of time-of-flight (ToF) sensors, and publishes both counts to an MQTT broker for use by another system.

This repository covers edge-side acquisition, counting, local image processing, and transmission. The camera count is a snapshot of visible people, while the ToF count is a cumulative carriage count based on doorway crossings.

## Repository overview

The main runner assembles the camera, shared carriage counter, optional ToF hardware, YOLO image processor, MQTT client, and monitoring service. Each monitoring cycle:

1. Captures a frame from the Raspberry Pi camera.
2. Reads the current doorway-based carriage count (ToF).
3. Runs local person detection on the frame.
4. Publishes the two counts and their source statuses as JSON over MQTT.

ToF polling runs continuously in a background thread.  
The camera and MQTT publishing cycle runs every 30 seconds in live mode.

## Main technologies

- **Python 3.12** (the repository pins `3.12.12` in `.python-version`)
- **Ultralytics YOLO** for person detection, with PyTorch model files and NCNN export artifacts included
- **OpenCV** and **NumPy** for image loading, enhancement, annotation, and storage
- **Paho MQTT** using MQTT 3.1.1 and QoS 1
- **Raspberry Pi camera tools** through the `rpicam-jpeg` command
- **Two VL53L0X ToF sensors** over I2C, controlled through Raspberry Pi GPIO and Adafruit CircuitPython libraries



## Main components

- **Monitor runner** (`scripts/monitor_runner.py`) configures the application, initializes its dependencies, starts ToF polling, and runs the processing loop.
- **Camera and image sources** (`src/hal/` and `src/sources/`) support Raspberry Pi cameras, V4L2 USB cameras, and images stored in `images/`.
- **Image processor** (`src/processing/image_processor.py`) runs a COCO-compatible YOLO model, retains class `0` (`person`) detections above the configured thresholds, and creates annotated images.
- **ToF doorway counter** (`src/hal/tof_unit.py`, `src/hal/tof_pair.py`, and `src/processing/carriage_counter.py`) polls two VL53L0X sensors and updates a thread-safe carriage count according to their trigger order.
- **Monitoring service** (`src/services/load_monitor_service.py`) combines image and sensor readings into one monitoring update while recording whether each source was available.
- **MQTT client and transport models** (`src/comms/`, `src/entities/`, `src/dto/`, and `src/converters/`) convert readings to the wire format and publish them.



## Data flow

The camera provides an image frame to `ImageProcessor`, which applies optional CLAHE contrast enhancement and counts valid person detections. Independently, the outside and inside ToF sensors are polled every 50 ms. An outside-to-inside sequence increments the shared `CarriageCounter`, the reverse sequence decrements it without allowing a negative count.

`LoadMonitorService` combines the latest camera count and ToF count with train and carriage identifiers, source statuses, and a timestamp. `MqttSensorClient` publishes the JSON payload to `train/sensors/updates` with these fields:

`trainId`, `carriageNumber`, `cameraCount`, `irCount`, `calculatedOccupancy`, `cameraStatus`, `irStatus`, and `timestamp`.



## Setup

### Requirements

- Raspberry Pi running Linux and Python 3.12
- An MQTT broker reachable from the Pi
- A COCO-compatible YOLO model whose class `0` is `person`
- One of:
  - A Raspberry Pi camera with `rpicam-jpeg` available
  - A V4L2-compatible USB camera
  - Images in `images/` for folder mode
- For doorway counting: two VL53L0X ToF sensors connected over I2C, with their XSHUT lines connected to GPIO 27 (outside) and GPIO 17 (inside)



### Install dependencies

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The runner also imports the Raspberry Pi GPIO/I2C and VL53L0X libraries, which are not currently declared in `requirements.txt`. Install them on the Pi:

```bash
python -m pip install adafruit-blinka adafruit-circuitpython-vl53l0x
```

Enable I2C in the Raspberry Pi system configuration before using the ToF sensors. Install the Raspberry Pi camera software separately if `rpicam-jpeg` is not already available.

### Configure

There is no environment-variable or external configuration-file support. Review the constants near the top of `scripts/monitor_runner.py` before running:

- `MQTT_BROKER` and `MQTT_PORT`
- `TRAIN_ID` and `CARRIAGE_NUMBER`
- `MODEL_PATH`
- Detection thresholds, image size, processing interval, and output directory

### Run

Run commands from the repository root so model, image, and output paths resolve correctly.

Raspberry Pi camera with ToF sensors:

```bash
python scripts/monitor_runner.py --mode live --camera rpi
```

USB camera:

```bash
python scripts/monitor_runner.py --mode live --camera usb
```

Process the repository's sample images without initializing ToF sensors:

```bash
python scripts/monitor_runner.py --mode images --no-sensors
```

The `--no-sensors` option skips ToF initialization and the runner still connects to MQTT during startup.

## Important code references

- `[TofPair](src/hal/tof_pair.py)` — Implements continuous two-sensor polling and the state machine that determines entry or exit from sensor trigger order.
- `[LoadMonitorService](src/services/load_monitor_service.py)` — Coordinates frame capture, count retrieval, person detection, status reporting, and transmission of each monitoring update.

## Repository structure

```text
.
├── images/                 # Sample carriage images
├── scripts/
│   ├── monitor_runner.py   # Application entry point and configuration
│   └── compare_models.py   # Model comparison utility
├── src/
│   ├── comms/              # MQTT publishing
│   ├── converters/         # Domain-to-wire conversion
│   ├── dto/                # MQTT payload data model
│   ├── entities/           # Frames, detections, readings, and sensor data
│   ├── hal/                # Cameras and VL53L0X hardware access
│   ├── interfaces/         # Camera, sensor, and communications contracts
│   ├── processing/         # Person detection and carriage counting
│   ├── services/           # Monitoring-cycle orchestration
│   ├── sources/            # Folder-backed image source
│   └── utils/              # Annotated-image output
├── yolo*.pt                # PyTorch YOLO model weights
├── .python-version
└── requirements.txt
```



## Current limitations

- Runtime settings are source constants rather than environment variables or a configuration file.
- The MQTT connection is unauthenticated and unencrypted in the current client.
- `requirements.txt` does not include the Raspberry Pi and ToF hardware libraries required by the runner imports.
- The carriage counter is held in memory and starts at zero whenever the process restarts.
- The application supports one configured ToF pair. The shared counter is designed for synchronized updates, but additional doors are not initialized by the current runner.
- `calculatedOccupancy` is transmitted as `0`, this edge application reports the camera and ToF counts separately.

