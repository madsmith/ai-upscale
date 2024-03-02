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

### Basic Usage ###

    python upscale-ai.py <input_files>

The tool will take the list of specified files and folders and iteratively attempt to apply an AI upscale to those files.  The default resolution is 720p but alternative resolutions can be specified with the ```-r``` flag.

#### Common Options ####

Rename the output file to indicate the quality setting with ```-n```.  This will either append ```-<resolution>``` to the video file or rewrite the ```[DVD]``` tag in the file name if present to ```[<resolution> AI Upscale]```.

Add black padding to the video file so that is a typical aspect ratio with the ```-p``` option.

Force over writing output files with ```-f```

#### Upscaler Profiles ####

Alternative upscaler profiles can be specified with the ```--upscaler-profile <profile-name>``` flag.  These allow alternative applications of AI models to the video footage.  Descriptions of the various AI models can be found [here](https://docs.topazlabs.com/video-ai/filters).

The default upscaler profile is ```iris-proteus```.  The iris AI model is trained to enhance facial details and the proteus model is a general purpose model for upscaling medium and high quality video.  This setting is setup for progressive video and the settings were deemed to be reasonable for a 1080p upscale of DVD content such as Deep Space 9.

Another setting is ```iris-artemis``` which aligns with the settings used by the original guide for upscalling Deep Space 9.

A third pre-made settings is ```proteus-5``` which is a 5% intensity application of the proteus AI model to the video source which seems to be adequate for upscaling animated progressive DVD content.

A dummy profile exists for testing purposes.  ```ffmpeg-scale``` will use ffmpeg's built-in scaling to scale the video without any AI enchancement of the video.

### Advanced Usage ###
#### Custom Profiles ####

To make the tool more flexible, upscale and encoding settings are stored in ```profiles.yml```.  Some default profiles are implemented but additional profiles can be added by creating new entries in the following format.

    profile-name:
      enable_builtin_filters: true
      ffmpeg_filter_complex: "<insert filter rules>"
      ffmpeg_metadata:
        upscaleai: "<Description to be written to videoai tag on output file>"

The encoding ffmpeg command from [Topaz Video AI] can be seen by right clicking on an export job and selecting ```ffmpeg command```.  That command looks like the following:

    ffmpeg "-hide_banner" "-t" "120.0199" "-i" "C:/Users/Martin/Projects/upconvert-ai/vfr_120.mp4" "-sws_flags" "spline+accurate_rnd+full_chroma_int" "-color_trc" "6" "-colorspace" "6" "-color_primaries" "6" "-filter_complex" "tvai_up=model=prob-4:scale=0:w=720:h=480:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=0:vram=1:instances=1,colorspace=ispace=5:space=6:primaries=6:trc=6" "-c:v" "hevc_nvenc" "-profile:v" "main" "-pix_fmt" "yuv420p" "-b_ref_mode" "disabled" "-tag:v" "hvc1" "-g" "30" "-preset" "p7" "-tune" "hq" "-rc" "constqp" "-qp" "17" "-rc-lookahead" "20" "-spatial_aq" "1" "-aq-strength" "15" "-b:v" "0" "-map" "0:a" "-map_metadata:s:a:0" "0:s:a:0" "-c:a" "copy" "-map_metadata" "0" "-map_metadata:s:v" "0:s:v" "-map" "0:s?" "-c:s" "copy" "-movflags" "use_metadata_tags+write_colr" "-metadata" "videoai=Enhanced using prob-4; mode: auto; revert compression at 0; recover details at 0; sharpen at 0; reduce noise at 0; dehalo at 0; anti-alias/deblur at 0; focus fix Off; and recover original detail at 20" "C:/Users/Martin/Projects/upconvert-ai/vfr_120_prob4.mkv"

Of particular note is the ```-filter_complex``` flag and the ```-metadata``` flag.

    <......>
    "-filter_complex" "tvai_up=model=prob-4:scale=0:w=720:h=480:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=0:vram=1:instances=1,colorspace=ispace=5:space=6:primaries=6:trc=6"
    <......>
    "-metadata" "videoai=Enhanced using prob-4; mode: auto; revert compression at 0; recover details at 0; sharpen at 0; reduce noise at 0; dehalo at 0; anti-alias/deblur at 0; focus fix Off; and recover original detail at 20"
    <......>

The values of these options are used to fill in ```ffmpeg_filter_complex``` and ```ffmpeg_metadata```.  Within the ```ffmpeg_filter_complex``` string, any substrings that appear in the format ```w=<number>``` or ```h=<number>``` are automatically updated and filled in with the target resolution.  The yaml key/values in ```ffmpeg_metadata``` are added as global metadata tags in the resulting video file.

The option ```enable_builtin_filters``` permits the tool to add generic padding, sample aspect ratio, color space corrections to the video file.  When enabled, parameters like:

    colorspace=ispace=5:space=6:primaries=6:trc=6
do not need to be specified in the ```ffmpeg_filter_complex``` setting, leaving (in this example) the ```ffmpeg_filter_complex``` setting equal to

    tvai_up=model=prob-4:scale=0:w=720:h=480:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=0:vram=1:instances=1

(**Note** the removal of the trailing ```,``` that separates filter chain rule groups)

#### Detailed Usage ####

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

