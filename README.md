# AI Upscaler

This project is based on the guide [DS9 Upscale Project][] by [Queerworm][].

[DS9 Upscale Project]: https://github.com/queerworm/ds9-upscale
[Queerworm]: https://github.com/queerworm

This project, like the [DS9 Upscale Project][], requires and leverages a software tool by [Topaz Labs][] called [Topaz Video AI][] for applying an upscale model.

The tools and processes described in the previous guides have been encapsulated into an all in one tool, allowing for automatic processing of files.

[Topaz Labs]: https://www.topazlabs.com
[Topaz Video AI]: https://www.topazlabs.com/topaz-video-ai

## Compatibility

This tool is currently compatible with Windows only.  While Topaz Video AI has a MacOS and Linux release, I have yet to attempt to port this tool to handle OS peculiarities.

The ffmpeg encoding settings are tuned for an NVidia GPU for hardware encoding.  See Topaz Video AI [command line documentation][] for details on encoding settings to tweak for other encoding environments and apply those settings by altering the profiles.yml file.

[command line documentation]: https://docs.topazlabs.com/video-ai/advanced-functions-in-topaz-video-ai/command-line-interface

## Dependencies

 * [Topaz Video AI]
 * [MKVToolNix] (mkvextract / mkvmerge)
 * [Python]
 * Python YAML module

## Installation

Install [Topaz Video AI]. This tool assumes the default install path is ```C:\Program Files\Topaz Labs LLC\Topaz Video AI``` and that the data files for the AI models is set to ```C:\ProgramData\Topaz Labs LLC\Topaz Video AI```

Run Topaz Video AI and login to your account.

**Note:** the original documentation noted that Topaz Video AI could be used in a trial mode.  I have not tested this work flow.

### Install MKVToolNix.

Download [MKVToolNix].  Run the installer.  This tool assumes MKVToolNix is installed at ```C:\Program Files\MKVToolNix\```

This tool is required for repairing variable frame rate video files.  The timestamps are extracted from the original video file and merged back into the final output file.  If the input file is detected to be constant frame rate, then this step will be skipped.

[MKVToolNix]: https://mkvtoolnix.download/

### Install Python ###

Download [python].

This tool was built and runs on python 3.10.  Python should be available in your environment (Command Prompt / PowerShell).  System environment setup is not covered by this documentation.

### Python Modules ###

The code uses the module [PyYAML].

    python3 -m venv myenv
    .\myenv\Scripts\activate
    pip install -r requirements.txt

Creating a Virtual Env and installing dependencies in that venv can prevent conflicts with other applications.  Activate your virtual environment when running python with ```.\myenv\Scripts\activate```

[python]: https://www.python.org/
[PyYAML]: https://pypi.org/project/PyYAML/

## Usage

Basic Usage of the tool

    python upscale-ai.py <input_files>

The tool will take the list of specified files and folders and iteratively attempt to apply an AI upscale to those files.  The default resolution is 720p but alternative resolutions can be specified with the ```-r``` flag.

Detailed Usage

    python upscale-ai.py -h
    usage: upscale-ai.py [-h] [-f] [-d] [--debug DEBUG] [--ext EXTENSION] [--open] [-o OUTPUT_DIR] [-n] [-r {1x,2x,4x,720,1080,1440,2160}] [--sar {source,square}]
                        [-s] [-p] [--upscaler {artemis,proteus}] [--upscaler-profile UPSCALER_PROFILE] [--ff:qp FF_QP] [-tvai TVAI_PATH] [-tvaidata TVAI_DATA_PATH]
                        [--mkvtoolnix MKVTOOLNIX_PATH]
                        [files ...]

    Upconvert Video

    positional arguments:
      files                 List of files and folders to process

    options:
      -h, --help            show this help message and exit
      -f, --force           Force reprocessing of files
      -d                    Enable debug output (debug level 1)
      --debug DEBUG         Enable debug output at a specific verbosity
      --ext EXTENSION, --extension EXTENSION
                            File extensions to process [multiple extensions seperated by ',']
      --open                Open the output file on completion
      -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                            Output directory for processed files

    Output Settings:
      -n, --rename          Rename output files to indicate the upscale resolution
      -r {1x,2x,4x,720,1080,1440,2160}, --resolution {1x,2x,4x,720,1080,1440,2160}
                            Resolution to target upscale [480, 720, 960, 1080, 1440, 2160, default=720]
      --sar {source,square}
                            Set Source Aspect Ratio [source, square, default=square]
      -s, --subfix          Fix subtitle rip issues, prefer SRT over VobSub and no default English track
      -p, --padding         Pad out video file to standard aspect ratio
      --upscaler {artemis,proteus}
                            Select upscaler model [artemis, proteus, default=proteus]
      --upscaler-profile UPSCALER_PROFILE
                            Select the upscaler profile from which to use the ffmpeg filter complex settings (see profiles.yml) [default=default]
      --ff:qp FF_QP         FFMPEG QP value for output video [default=17]

    Path Overrides:
      -tvai TVAI_PATH, --topaz-video-ai TVAI_PATH
                            Path to Topaz Video Enhance AI installation
      -tvaidata TVAI_DATA_PATH, --topaz-video-ai-data TVAI_DATA_PATH
                            Path to Topaz Video Enhance AI installation Data Directory
      --mkvtoolnix MKVTOOLNIX_PATH
                            Path to mkvtoolnix installation where mkvextract is located

