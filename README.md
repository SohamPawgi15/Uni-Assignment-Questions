For Q.2 of 1.py assignment  questions of Zone Plate Animation

## Requirements

- `numpy`
- `matplotlib`
- `ffmpeg`

## Setup

### Install Python Packages

You can install the required Python packages using `pip`:

```sh
pip install numpy matplotlib
```

### Install `ffmpeg`

1. Download `ffmpeg` from the [official website](https://ffmpeg.org/download.html).
2. Follow the instructions to install `ffmpeg` on your system.
3. Ensure that `ffmpeg` is added to your system's PATH, or set the path in the script using the `FFMPEG_PATH` environment variable.

### Set `ffmpeg` Path

If `ffmpeg` is not in your system's PATH, you can set the path in the script using an environment variable:

#### On Windows
```sh
setx FFMPEG_PATH "C:\path\to\ffmpeg\bin\ffmpeg.exe"
```

#### On macOS/Linux
```sh
export FFMPEG_PATH=/path/to/ffmpeg/bin/ffmpeg
```

Replace `/path/to/ffmpeg/bin/ffmpeg` with the actual path to the `ffmpeg` executable.

### Project Structure

Ensure your project directory has the following structure:

```
Project/
│
├── script.py
├── images/
│   └── elvis.bmp
└── README.md
```

### Running the Script

You can run the script using Python:

```sh
python script.py
```

This will generate a `zoneplate.mov` file with the animation.

Enjoy generating zone plate animations!