#!/usr/bin/env python
from dataclasses import dataclass
import shutil
import subprocess
import concurrent.futures
import threading
import json
import os
import re
from pathlib import Path
import time
import argparse
import traceback

import yaml
from util import Timestamp

LVL_INFO = 1
LVL_DEBUG = 2
LVL_TRACE = 3
LVL_DUMP = 4

@dataclass
class Resolution:
    width: int
    height: int

    def __getitem__(self, x):
        if x == 'w': return self.width
        if x == 'h': return self.height
        if x == 0: return self.width
        if x == 1: return self.height
        return None

    def __getattr__(self, name):
        if name == 'w':
            return self.width
        elif name == 'h':
            return self.height
        else:
            # Raise AttributeError to comply with Python's expected behavior
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __mul__(self, other):
        return Resolution(self.width * other, self.height * other)

    def __rmul__(self, other):
        # This ensures multiplication works in both orders: scalar * Resolution and Resolution * scalar
        return self.__mul__(other)

    def __str__(self):
        return f"{self.width}x{self.height}"

class AspectRatio:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self._divisor = self._gcd(self.width, self.height)

    @classmethod
    def from_string(cls, value):
        width, height = map(int, value.split(":"))
        return cls(width, height)

    @classmethod
    def from_resolution(cls, resolution):
        return cls(resolution.width, resolution.height)

    def _gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def value(self):
        return self.width / self.height

    # Allow aspect ratios to be divided by each other
    def __truediv__(self, other):
        if isinstance(other, AspectRatio):
            return self.value() / other.value()
        return NotImplemented

    def __str__(self):
        return f"{(int)(self.width/self._divisor)}:{int(self.height/self._divisor)}"

class Packet:
    def __init__(self, probe_data, time_base):
        self.data = probe_data
        # Convert time_base 1/1000 to a tuple (1, 1000)
        self.time_base = tuple(map(int, time_base.split("/")))

    def timestamp(self):
        # Prefer pkt_dts, then best_effort_timestamp, then pts
        if 'dts' in self.data:
            return self._timestamp_to_ms(self.data["dts"])
        if 'best_effort_timestamp' in self.data:
            return self._timestamp_to_ms(self.data["best_effort_timestamp"])
        if 'pts' in self.data:
            return self._timestamp_to_ms(self.data["pts"])
        return None

    def time(self):
        # Prefer pkt_dts, then best_effort_timestamp, then pts
        if 'dts_time' in self.data:
            return float(self.data["dts_time"])
        if 'best_effort_timestamp_time' in self.data:
            return float(self.data["best_effort_timestamp_time"])
        if 'pts_time' in self.data:
            return float(self.data["pts_time"])
        return None

    def _timestamp_to_ms(self, timestamp):
        return (timestamp * self.time_base[0]) / self.time_base[1] * 1000

    def media_type(self):
        return self.data.get("media_type")

    def stream_index(self):
        return self.data.get("stream_index")

    # Render
    def __str__(self):
        # Select best_effort_timestamp, pkt_dts, and pts and media_type and stream_index if present
        print_data = {k: self.data[k] for k in [
            "media_type", "stream_index",
            "pts", "pkt_dts", "best_effort_timestamp",
            "pts_time", "pkt_dts_time", "best_effort_timestamp_time"
            ] if k in self.data}
        return json.dumps(print_data, indent=2)

class UpscaleJob:
    """A class to represent a conversion job."""

    def __init__(self, file):
        self.file = file
        self.filename = file.name
        self.working_dir = get_working_directory(self.filename)
        self.output_file = self._determine_output_file()
        self.intermediate_file = None
        self.streams = []
        self.timestamps = None
        self.frame_count = None
        self.probe_data = None

    def _determine_output_file(self):
        if get_globals("rename"):
            extension = self.file.suffix
            resolution = get_globals("resolution")
            # If file contains [] tag, then replace than tag appropriately
            tag = re.match(r".*\[([^\]]+)\][^\[]*", self.filename)
            if tag:
                # Replace DVD or DVDRip with resolution
                quality_tag = f"{resolution}p AI Upscale"
                new_tag = re.sub(r"DVD(Rip)?", quality_tag, tag.group(1), flags=re.IGNORECASE)
                return get_output_directory() / f"{self.filename.replace(tag.group(1), new_tag)}"
            return get_output_directory() / f"{self.filename.replace(extension, f'-{resolution}{extension}')}"
        return get_output_directory() / self.filename

    def process(self):
        log_job(f"Processing file: {str(self.file)}")

        self.frame_count = self.estimate_frame_count(self.file)
        # Check if the output file already exists

        if self.output_file.exists() and not get_globals("force"):
            if self.output_file.stat().st_size == 0:
                remove_file(self.output_file)
            else:
                output_file_frame_count = self.estimate_frame_count(self.output_file)

                # There may be a small deviation of the output frame count due to TVAI filters
                if abs(output_file_frame_count - self.frame_count) > 2:
                    log_step(f"Output file frame count mismatch, removing output file")
                    remove_file(self.output_file)
                else:
                    log_step(f"Skipping Completed Job: {str(self.output_file)}")
                    return


        # Analyze the video file
        self.analyze_video()

        # Check if the video is variable frame rate
        is_vfr = self.is_variable_frame_rate()
        if debug_check(LVL_DEBUG):
            print(f"Variable Frame Rate: {is_vfr}")

        if is_vfr:
            # Get the timestamps from the input file
            self.extract_input_timestamps()

        # Run upscale job to intermediate file
        self.upscale_ai()

        if is_vfr:
            # Merge the timestamps into the output file
            self.rebuild_timestamps()
        else:
            try:
                # Move the intermediate file to the output file
                self.intermediate_file.replace(self.output_file)
            except OSError as e:
                # Try to copy the file if the move fails
                shutil.copy2(self.intermediate_file, self.output_file)
                remove_file(self.intermediate_file)
                # Let potential exceptions bubble up

        self.remove_intermediates()

        log_step(f"Completed Job: {str(self.output_file)}")

        if get_globals("open"):
            os.startfile(str(self.output_file))

    def analyze_video(self):
        log_step("Analyzing video...")
        if self.probe_data is not None:
            return

        json_data = self._process_analyze_video()
        self.probe_data = json.loads(json_data)

        if debug_check(LVL_DUMP):
            print(json.dumps(self.probe_data, indent=2))

    def estimate_frame_count(self, file):
        frame_count = self._process_estimate_frame_count(file)
        # For some reason, ffprobe occaisionally returns a trailing | character
        # Strip newlines then any trailing | character
        frame_count = int(frame_count.strip().strip("|"))
        return frame_count

    def is_variable_frame_rate(self):
        if "packets" not in self.probe_data:
            raise Exception("No frame packets found in probe data")

        # Get the timebase from the first stream in streams with codec_type = video
        timebase = next((stream["time_base"] for stream in self.probe_data["streams"]
                         if stream["codec_type"] == "video"), None)
        video_packets = [Packet(frame, timebase) for frame in self.probe_data["packets"]
                        if frame["codec_type"] == "video" and frame['stream_index'] == 0]
        time_deltas = [video_packets[i+1].time() - video_packets[i].time()
                       for i in range(0, len(video_packets) - 1)]

        # Compute Mean and CV of time deltas
        mean = sum(time_deltas) / len(time_deltas) if len(time_deltas) > 0 else 0
        cv = (sum((x - mean) ** 2 for x in time_deltas) / len(time_deltas)) ** 0.5 / mean if len(time_deltas) > 0 else 0

        if debug_check(LVL_DEBUG):
            print(f"Is VFR: {cv > 0.05} Mean: {mean} CV: {cv}")
        return cv > 0.05

    def extract_input_timestamps(self):
        if self.timestamps is not None:
            return

        self.load_frame_data()

        log_step("Building timestamps...")
        frame_data = self.probe_data["frames"]
        self.timestamps = self._convert_to_timestamps(frame_data)

    def _convert_to_timestamps(self, frames):
        timestamps = []
        timebase = next((stream["time_base"] for stream in self.probe_data["streams"]
                         if stream["codec_type"] == "video"), None)

        if timebase is None:
            raise Exception("No video timebase found in probe data")

        timebase = tuple(map(int, timebase.split("/")))

        for frame in frames:
            time_ms = (frame["pts"] * 1000 * timebase[0]) / timebase[1]
            timestamps.append(Timestamp(time_ms))

        return timestamps

    def upscale_ai(self):
        log_step("Upscaling frames...")

        # Ensure the upscaled folder exists
        upscale_folder = self.working_dir / "upscaled"
        ensure_folder_exists(upscale_folder)

        self.intermediate_file = upscale_folder / self.filename

        if self.intermediate_file.exists() and not get_globals("force"):
            if self.intermediate_file.stat().st_size == 0:
                remove_file(self.intermediate_file)
            else:
                intermediate_frame_count = self.estimate_frame_count(self.intermediate_file)
                if abs(intermediate_frame_count - self.frame_count) > 2:
                    log_step("Intermediate file frame count mismatch, removing intermediate file")
                    remove_file(self.intermediate_file)
                else:
                    log_step("Intermediate file already exists")
                    return

        process = self._process_upscale_video(self.file, self.intermediate_file)

        # Monitor the upscale job
        print("      Job Starting...", end="\r", flush=True)
        job_status_pattern = re.compile(r"^frame=\s*(\d+)\s+fps= *([\d.]+) .*size= *([^ ]+)")
        last_line_length = 0
        job_output = []
        while True:
            output_line = process.stdout.readline()
            if not output_line:
                break
            match = job_status_pattern.search(output_line)
            if match:
                frames_processed = int(match.group(1))
                processing_fps = float(match.group(2))
                output_size = match.group(3)
                if processing_fps > 0:
                    time_remaining = (self.frame_count - frames_processed) / processing_fps
                else:
                    time_remaining = None
                progress = frames_processed / max(self.frame_count, frames_processed)

                render_line = f"      Time Remaining: {self._render_time(time_remaining)} [{progress:.2%}] - Frames: {frames_processed} FPS: {processing_fps} Size: {output_size}"
                new_line_length = len(render_line)
                # Pad render line to last line length
                render_line += " " * (last_line_length - new_line_length)
                print(render_line, end="\r", flush=True)
                last_line_length = new_line_length
            else:
                job_output.append(output_line)
                if debug_check(LVL_DEBUG):
                    print(f"{output_line}", end="")

        return_code = process.wait()

        # Clear the line
        print(" " * last_line_length, end="\r", flush=True)
        if return_code != 0:
            remove_file(self.intermediate_file)
            msg = ("Upscale Job Failed: [ffmpeg output]" + os.linesep + "  "
                + "  ".join(job_output[-10:]))
            raise Exception(msg)
        else:
            log_step("Upscale complete")

    def load_frame_data(self):
        if 'frames' in self.probe_data:
            return

        log_step("Loading video frame data...")

        # Load frame data from the intermediate file
        output = self._process_analyze_frame_data(self.file)
        frame_data = json.loads(output)["frames"]

        self.probe_data["frames"] = frame_data

    def rebuild_timestamps(self):
        log_step("Loading Intermediate Frame Data...")
        intermediate_frame_count = self.estimate_frame_count(self.intermediate_file)

        delta = intermediate_frame_count - len(self.timestamps)

        log_step("Rebuilding timestamps...")

        # Build the merge timestamps file
        merge_timestamps = self.working_dir / "merge_timestamps.txt"
        with merge_timestamps.open("w") as file:
            file.write("# timestamp format v2\n")
            for ts in self.timestamps:
                file.write(ts.getValueStr() + "\n")
            # Pad out timestamps to match intermediate file
            if delta > 0:
                for i in range(0, delta):
                    file.write(self.timestamps[-1].getValueStr() + "\n")

        # Merge the timestamps into the output file
        self._process_merge_timestamps(self.output_file, self.intermediate_file, merge_timestamps)
        remove_file(merge_timestamps)

    def remove_intermediates(self):
        log_step("Removing intermediate files...")

        # Remove intermediate video
        remove_file(self.intermediate_file)
        remove_dir(self.working_dir / "upscaled")
        remove_dir(self.working_dir)

    def _process_run(self, command, parallelSafe=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=None):
        if debug_check(LVL_DEBUG):
            print(" ".join(shell_escape(str(c)) for c in command))
        if env is None:
            env = os.environ.copy()
        if parallelSafe: # or command[0] == "ffmpeg":
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, env=env)
        else:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)

        if process.returncode != 0:
            process_output = process.stdout.decode('utf-8')
            process_error = process.stderr.decode('utf-8')
            output = process_error if process_error else process_output
            raise Exception(f"Command failed: {' '.join(shell_escape(c) for c in command)}\n{output}")

        return process.stdout.decode('utf-8')

    def _process_analyze_video(self):
        command = [get_ffprobe(), "-hide_banner", "-loglevel", "0", "-of", "json",
            "-show_format", "-show_streams", "-show_chapters", "-show_packets", self.file]
        return self._process_run(command)

    def _process_analyze_frame_data(self, file):
        command = [get_ffprobe(), "-hide_banner", "-loglevel", "0", "-select_streams", "v:0",
                   "-show_frames", "-of", "json", file]
        return self._process_run(command)

    def _process_estimate_frame_count(self, file):
        command = [get_ffprobe(), "-hide_banner", "-loglevel", "0", "-select_streams", "v:0",
                   "-count_packets", "-show_entries", "stream=nb_read_packets",
                   "-of", "compact=p=0:nk=1", file]
        return self._process_run(command)

    # def _process_extract_timestamp(self, input_file, timestamp_file):
    #     command = [get_mkvextract(), input_file, "timestamps_v2", "0:" + str(timestamp_file)]
    #     return self._process_run(command)

    def _get_stream_abbreviation(self, stream_data):
        if stream_data['codec_type'] == "video":
            return "v"
        if stream_data['codec_type'] == "audio":
            return "a"
        if stream_data['codec_type'] == "subtitle":
            return "s"
        if stream_data['codec_type'] == "data":
            return "d"
        if stream_data['codec_type'] == "attachment":
            return "t"

        return None

    # Upscale Video Process and Helpers
    def _build_ffmpeg_streams(self):
        input_stream = 0
        output_state = {}

        # Map all global metadata from the input file
        stream_options = ["-map_metadata", f'{input_stream}']

        # Pretty print json probe_data
        #print(json.dumps(self.probe_data["streams"], indent=2))

        # TODO: can't copy mov_text subtitles to mkv
        streams = self.probe_data["streams"]
        subtitle_streams = [stream for stream in streams if stream['codec_type'] == "subtitle"]
        subtitle_stream_in_map = {stream['index']: pos for pos, stream in enumerate(subtitle_streams)}
        if get_globals("subfix"):
            subtitle_streams.sort(key=lambda x: (x['codec_name'] == "dvd_subtitle", x['index']))
            streams.sort(key=lambda x: (x['codec_name'] == "dvd_subtitle", x['index']))
        subtitle_stream_out_map = {stream['index']: pos for pos, stream in enumerate(subtitle_streams)}

        for stream in streams:
            type_abbr = self._get_stream_abbreviation(stream)

            if type_abbr and type_abbr in ["v", "a", "s"]:
                # Assuming a single video stream
                if type_abbr == "v":
                    # Reference filtered stream as [v]
                    stream_options.extend(["-map", "[v]", "-map_metadata:s:v", f"{input_stream}:s:{type_abbr}"])

                    # Output will be MKV which is MOV/MP4 compatible
                    # Set flags:
                    #  faststart - move the moov atom to the beginning of the file
                    #  use_metadata_tags - use the mtda atom for metadata
                    #  write_colr - write the colr atom to the output file
                    stream_options += ["-movflags", "faststart+use_metadata_tags+write_colr"]
                elif type_abbr == "s" or type_abbr == "a":
                    if not type_abbr in output_state:
                        output_state[type_abbr] = {'in': 0, 'out': 0}

                        # Map all stream to output
                        if get_globals("subfix") and type_abbr == "s":
                            for sid, pos in subtitle_stream_out_map.items():
                                src_pos = subtitle_stream_in_map[sid]
                                stream_options.extend(['-map', f'0:{type_abbr}:{src_pos}'])
                        else:
                            stream_options.extend(['-map', f'0:{type_abbr}'])

                        # Copy streams without conversion
                        if type_abbr != "v": # Currently redundant
                            stream_options.extend([f'-c:{type_abbr}', 'copy'])

                    out_idx = output_state[type_abbr]['out']
                    in_idx = output_state[type_abbr]['in']

                    if g_globals["subfix"] and type_abbr == "s":
                        out_idx = subtitle_stream_out_map[stream['index']]
                        in_idx = subtitle_stream_in_map[stream['index']]

                    # Map metadata for the stream to the corresponding output stream
                    output_spec = f's:{type_abbr}:{out_idx}'
                    input_spec = f's:{type_abbr}:{in_idx}'
                    stream_options.extend([f"-map_metadata:{output_spec}", f"{input_stream}:{input_spec}"])

                    if type_abbr == "s" and get_globals("subfix"):
                        if stream["disposition"] and stream["disposition"]["default"]:
                            stream_options.extend([f"-disposition:s:{out_idx}", "0"])

                    # Output stream was mapped, increment the output stream index
                    output_state[type_abbr]['out'] += 1
                    # Increment the input stream index
                    output_state[type_abbr]['in'] += 1

        return stream_options

    def _get_output_resolution(self):
        video_stream = next((stream for stream in self.probe_data["streams"] if stream["codec_type"] == "video"), None)
        if video_stream is None:
            raise Exception("No video stream found")

        current_resolution = Resolution(video_stream["width"], video_stream["height"])
        target_resolution = current_resolution
        dar = AspectRatio.from_string(video_stream["display_aspect_ratio"])
        sar = AspectRatio.from_string(video_stream["sample_aspect_ratio"])

        standard_resolutions = {
            "720":  Resolution(1280, 720),
            "1080": Resolution(1920, 1080),
            "1440": Resolution(2560, 1440),
            "2160": Resolution(3840, 2160)
        }

        resolution_tag = get_globals("resolution")

        if debug_check(LVL_DEBUG):
            print(f"Video Stream: {json.dumps(self.probe_data['streams'][0], indent=2)}")

        # Apply resolution scalar to target resolution
        scalers = {"1x": 1, "2x": 2, "4x": 4}
        if resolution_tag in scalers:
            if get_globals("pix_fmt") == "square":
                current_resolution = Resolution(int(round(current_resolution.width * sar.value())),
                                                current_resolution.height)

            target_resolution = current_resolution * scalers[resolution_tag]

        # Lookup standard Resolution
        elif resolution_tag in standard_resolutions:
            target_resolution = standard_resolutions.get(resolution_tag)
        else:
            raise Exception(f"Unsupported resolution: {resolution_tag}")

        target_resolution_aspect_ratio = AspectRatio.from_resolution(target_resolution)
        if debug_check(LVL_DEBUG):
            print(f"Initial Target Resolution: {target_resolution} @ {target_resolution_aspect_ratio}")
            print(f"Source Resolution: {current_resolution} @ {dar}")

        # If padding is disabled and resolution isn't a "close" match, attempt to adjust width to match display resolution
        if not get_globals("padding") and (target_resolution_aspect_ratio / dar) > 1.01:
            # Compute appropriate width for target resolution
            new_width = int(round(target_resolution[1] * dar.value()))
            # Round up to nearest multiple of 4
            new_width = (new_width + 3) & ~3
            target_resolution = Resolution(new_width, target_resolution[1])
            target_resolution_aspect_ratio = AspectRatio.from_resolution(target_resolution)

        if debug_check(LVL_DEBUG):
            print(f"Adjusted Target Resolution: {target_resolution} @ {AspectRatio.from_resolution(target_resolution)}")
            print(f"Target Resolution Aspect Ratio: {target_resolution_aspect_ratio} Display Aspect Ratio: {dar} @ {target_resolution_aspect_ratio / dar}")

        if (get_globals("pix_fmt") == "square"):
            if (target_resolution_aspect_ratio / dar) < 1.01 or get_globals("padding"):
                return target_resolution
            else:
                raise Exception(f"Display aspect ratio mismatch: {dar} vs {target_resolution_aspect_ratio} @ {target_resolution}")
        else:
            return target_resolution

    def _get_input_display_resolutions(self):
        video_stream = next((stream for stream in self.probe_data["streams"] if stream["codec_type"] == "video"), None)
        if video_stream is None:
            raise Exception("No video stream found")

        current_resolution = Resolution(video_stream["width"], video_stream["height"])

        # Scale width and height to display aspect ratio
        sar = AspectRatio.from_string(video_stream["sample_aspect_ratio"])
        display_width = int(round(current_resolution.width * sar.value()))
        display_resolution = Resolution(display_width, current_resolution.height)

        if debug_check(LVL_DEBUG):
            print(f"Video Resolution: {current_resolution}")
            print(f"SAR: {sar}")
            print(f"Display Resolution: {display_resolution}")

        return (current_resolution, display_resolution)

    def _load_profile_data(self, profile_name):
        if profile_name == "default":
            return None

        if not hasattr(self, "profiles"):
            self.profiles = {}
            profile_yaml_file = get_profile_yaml()
            with open(profile_yaml_file) as file:
                yaml_data = yaml.safe_load(file)
                if 'profiles' in yaml_data:
                    self.profiles = yaml_data['profiles']
                else:
                    raise Exception(f"No profiles found in {profile_yaml_file}")
            if debug_check(LVL_TRACE):
                print(f"Current Profiles: {self.profiles}")

        profile = self.profiles.get(profile_name)
        if profile is None:
            raise Exception(f"Profile not found: {profile_name}")
        profile['name'] = profile_name
        return profile

    def _load_upscaler_profile(self, profile, output_resolution):
        filter_str = profile.get("ffmpeg_filter_complex")
        if filter_str is None:
            raise Exception(f"Profile missing ffmpeg_filter_complex: {profile['name']}")

        # Replace resolution variables
        def replace_resolution(match):
            return match.group(1) + "=" + str(output_resolution[match.group(1)])
        filter_str = re.sub(r'([wh])=(\d+)', replace_resolution, filter_str)
        return filter_str

    def _load_upscaler_profile_metadata(self, profile):
        metadata = profile.get("ffmpeg_metadata", {})
        metadata_options = []
        for k, v in metadata.items():
            metadata_options.extend(["-metadata", f"{k}={v}"])
        return metadata_options

    def _process_upscale_video(self, input_file, output_file):
        env = os.environ.copy()
        env['TVAI_MODEL_DATA_DIR'] = str(get_tvai_data_path())
        env['TVAI_MODEL_DIR'] = str(get_tvai_data_path() / "models")

        output_resolution = self._get_output_resolution()
        original_resolution, original_display_resolution = self._get_input_display_resolutions()

        ffmpeg_options = ["-hide_banner", "-y"]
        color_options = ["-color_trc", "2", "-colorspace", "0", "-color_primaries", "6"]
        scalar_options = ["-sws_flags", "spline+accurate_rnd+full_chroma_int+full_chroma_inp"]
        # TODO: is this clip variable frame rate, compare frames to timestamp * fps
        fps_options = ["-fps_mode:v", "vfr"]
        input_options = ["-i", input_file]
        output_options = [output_file]

        # Prescale to display resolution
        filter_prescale = ""
        if g_globals["pix_fmt"] == "square":
            filter_prescale = f"scale=w={original_display_resolution.width}:h={original_display_resolution.height},setsar=1"

        # The below settings are specific for an NVidia GPU.  Topaz Labs documents
        # settings to add/remove for compatibility with other GPUs (Intel, AMD, Apple Silicon, etc.)
        # If you're system differs from this setup, you may need to adjust these settings.

        filter_str = ""
        profile_name = get_globals("upscaler_profile")
        profile = self._load_profile_data(profile_name)
        if profile_name != "default":
            filter_str = self._load_upscaler_profile(profile, output_resolution)
            if profile.get("enable_builtin_filters", False):
                filters = [filter_prescale, filter_str]

                # If output size differs from input size, add post-scaling filter
                if output_resolution != original_resolution:
                    filters.append(f"scale=w={output_resolution.w}:h={output_resolution.h}:flags=lanczos:threads=0:force_original_aspect_ratio=decrease")

                # If padding is enabled, add padding filter
                if get_globals("padding") and not get_globals("resolution") in ["1x", "2x", "4x"]:
                    # TODO: We only need to pad if the aspect ratio changes
                    filters.append(f"pad={output_resolution.w}:{output_resolution.h}:-1:-1:color=black")

                filters.append("colorspace=ispace=5:space=6:primaries=6:trc=6")
                # Add built-in filters
                filter_str = ','.join([x for x in filters if x])
            else:
                filter_str = False

        if get_globals("upscaler_profile") == "default" or filter_str == None:
            # Build Topaz Video AI filter
            filter_iris = "tvai_up=model=iris-2:scale=1:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=0:vram=1:instances=1"
            # TODO: factor out resolution
            filter_artemis = f"tvai_up=model=amq-13:scale=0:w={output_resolution.w}:h={output_resolution.h}:blend=0.2:device=0:vram=1:instances=1"
            filter_proteus = f"tvai_up=model=prob-4:scale=0:w={output_resolution.w}:h={output_resolution.h}:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=0:vram=1:instances=1"
            filter_upscaller = filter_proteus
            if get_globals("upscaler") == "artemis":
                filter_upscaller = filter_artemis
            filter_scale = f"scale=w={output_resolution.w}:h={output_resolution.h}:flags=lanczos:threads=0:force_original_aspect_ratio=decrease"
            filter_pad = ""
            if get_globals("padding") and not get_globals("resolution") in ["1x", "2x", "4x"]:
                filter_pad = f"pad={output_resolution.w}:{output_resolution.h}:-1:-1:color=black"
            filter_color = "colorspace=ispace=5:space=6:primaries=6:trc=6"
            filter_str = ",".join([x for x in [
                    filter_prescale, filter_iris, filter_upscaller, filter_scale, filter_pad , filter_color
                ] if x])
        if filter_str == False:
            raise Exception(f"Profile {profile_name} is missing ffmpeg_filter_complex")
        else:
            filter_options = ["-filter_complex", f"[0:v]{filter_str}[v]"]

        stream_options = self._build_ffmpeg_streams()
        video_encode_options = ["-c:v", "hevc_nvenc", "-profile:v", "main", "-pix_fmt", "yuv420p", "-b_ref_mode",
                                 "disabled", "-tag:v", "hvc1", "-g", "30", "-preset", "p7", "-tune", "hq",
                                 "-rc", "constqp", "-qp", get_globals("ff_qp"), "-rc-lookahead", "20", "-spatial_aq", "1",
                                 "-aq-strength", "15", "-b:v", "0"]

        metadata_options = []
        if profile_name != "default":
            metadata_options = self._load_upscaler_profile_metadata(profile)
        else:
            # Add metadata indicating how the file has been manipulated
            iris_desc = "Enhanced using iris-2; mode: auto; revert compression at 0; recover details at 0; sharpen at 0; reduce noise at 0; dehalo at 0; anti-alias/deblur at 0; focus fix Off; and recover original detail at 20."
            amq_desc = "Enhanced using amq-13; and recover original detail at 20."
            proteus_desc = "Enhanced using prob-4; mode: auto; revert compression at 0; recover details at 0; sharpen at 0; reduce noise at 0; dehalo at 0; anti-alias/deblur at 0; focus fix Off; and recover original detail at 20."
            scaler_desc = proteus_desc
            if get_globals("upscaler") == "artemis":
                scaler_desc = amq_desc
            metadata_desc = f"{iris_desc} {scaler_desc}"
            metadata_options = ["-metadata", f"videoai={metadata_desc}"]

        if (profile and profile.get("enable_builtin_filters", False)) or profile_name == 'default':
            if (original_resolution != output_resolution):
                resolution_desc = f" Changed resolution from {original_resolution} to {output_resolution}."
                # Find '-metadata' entry in metadata options, and add text to next entry in metadata_options
                for i, v in enumerate(metadata_options):
                    if i % 2 == 0:
                        continue
                    if v.startswith("videoai="):
                        metadata_options[i] += resolution_desc
                    if v == '-metadata':
                        metadata_options.insert(i+1, "resolution_change=true")
                        break



        command = ([get_ffmpeg()] + ffmpeg_options + input_options + color_options + scalar_options
                   + fps_options + filter_options + stream_options + video_encode_options + metadata_options
                   + output_options)

        # Dump Environment and Command for debugging
        # for (k, v) in env.items():
        #     print(f"{k}={shell_escape(v)}")
        if debug_check(LVL_INFO):
            print(" ".join(shell_escape(str(c)) for c in command))
        #return self._process_run(command, env=env)
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, universal_newlines=True)

    def _process_merge_timestamps(self, output_file, input_file, timestamp_file):
        command = [get_mkvmerge(), "-o", output_file, "--timestamps", f"0:{timestamp_file}", input_file]
        return self._process_run(command)

    # DEAD CODE
    # def extract_frames(self):
    #     log_step("Extracting frames...")

    #     # Ensure the frames folder exists
    #     self.frames_folder = ensure_folder_exists(self.working_dir / "frames")

    #     self._extract_frames()

    #     log_step("All frames extracted")

    # def _extract_frames(self):
    #     extract_finished = threading.Event()
    #     extract_thread = threading.Thread(target=self._extract_all_frames)
    #     extract_thread.start()

    #     progress_thread = threading.Thread(target=self._extract_frames_progress, args=(extract_finished,))
    #     progress_thread.start()

    #     extract_thread.join()
    #     extract_finished.set()
    #     progress_thread.join()

    # def _extract_frames_progress(self, event):
    #     counter = 0
    #     while not event.is_set():
    #         tok = {0: "-", 1: "\\", 2: "|", 3: "/"}.get(counter % 4)
    #         current_count = len(list(self.frames_folder.glob("*.png")))
    #         print(f"\r    [{tok}] {current_count:6} / {self.frame_count} frames extracted", end="")
    #         time.sleep(0.100)  # Check every second
    #         counter += 1
    #     print("\r" + " " * 78 + "\r", end="")

    # def _extract_frames_parallel(self):
    #     progress = [0, 0, 0]
    #     def update_progress(progress):
    #         progress[0] += 1

    #         tok = {0: "-", 1: "\\", 2: "|", 3: "/"}.get(progress[2] % 4)
    #         print(f"\r    [{tok}] {progress[0]:6} / {self.frame_count} frames extracted", end="")

    #         if time.time() - progress[1] > 0.100:
    #             progress[1] = time.time()
    #             progress[2] += 1

    #     # We have to have ffmpeg extract frames one at a time because we're seeking to specific timestamps to grab the frame
    #     # since our input is a variable frame rate video.
    #     with concurrent.futures.ProcessPoolExecutor(initializer=init_globals, initargs=get_globals()) as executor:
    #         futures = {executor.submit(self._extract_single_frame, (frame)) for frame in range(0, self.frame_count)}

    #         for future in concurrent.futures.as_completed(futures):
    #             if future.exception() is not None:
    #                 raise future.exception()

    #             update_progress(progress)

    #     print("\r" + " " * 56 + "\r", end="")

    # def _extract_all_frames(self):
    #     output_file_pattern = self.frames_folder / "frame_%06d.png"
    #     self._process_extract_all_frames(output_file_pattern)

    # def _extract_single_frame(self, i):
    #     try:
    #         output_frame = self.frames_folder / f"frame_{(i+1):06d}.png"
    #         self._process_extract_frame(output_frame, self.timestamps[i])

    #         # Validate that the frame was extracted
    #         if not output_frame.exists():
    #             raise Exception(f"Frame not extracted: {output_frame} @ {self.timestamps[i]}")
    #     except Exception as e:
    #         print(e)

    # def _process_extract_all_frames(self, output_file_pattern):
    #     command = [get_ffmpeg(), "-y", "-i", self.file, "-q:v", "2", "-vsync", "0", "-v", "fatal", output_file_pattern]
    #     return self._process_run(command)

    # def _process_extract_frame(self, output_frame, timestamp):
    #     command = [get_ffmpeg(), "-y", "-i", self.file, "-ss", timestamp, "-vframes", "1", "-q:v", "2", output_frame]
    #     return self._process_run(command)

    # def _process_do_nop(self):
    #     command = [get_ffmpeg()]
    #     return self._process_run(command, parallelSafe=False)

    def _render_time(self, seconds):
        if seconds is None:
            return "N/A"
        seconds = int(seconds)

        # Simplify time rendering if hours or minutes are not present
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return time.strftime("%M:%S", time.gmtime(seconds))
        else:
            return time.strftime("%H:%M:%S", time.gmtime(seconds))

def shell_escape(value):
    str_value = str(value)
    special_chars = ' &|()^,;%!'
    return f'"{str_value}"' if any(c in str_value for c in special_chars) else str_value

def log_job(message):
    log_n(0, message)

def log_step(message):
    log_n(1, message)

def log_substep(message):
    log_n(2, message)

def log_n(n, message):
    print("    " * (n-1) + "  - " * (1 if n > 0 else 0) + message)

def ensure_folder_exists(folder):
    folder = Path.cwd() / folder
    if not folder.exists():
        folder.mkdir(parents=True)

    if not folder.is_dir():
        raise Exception(f"Unable to created needed directory: {folder}")

    return folder

def get_working_directory(subpath=None):
    tmp_folder = Path.cwd() / "tmp"
    if subpath is not None:
        tmp_folder = tmp_folder / subpath
    ensure_folder_exists(tmp_folder)
    return tmp_folder

def get_output_directory():
    return ensure_folder_exists(get_globals("output_dir"))

def remove_file(file):
    if file.exists():
        try:
            file.unlink()
        except Exception as e:
            print(f"Unable to remove file: {file}")
            print(e)

def remove_dir(dir):
    if dir.exists():
        try:
            dir.rmdir()
        except Exception as e:
            print(f"Unable to remove directory: {dir}")
            print(e)


def process_file(file):
    job = UpscaleJob(file)
    try:
        job.process()
    except Exception as e:
        print(f"Error processing file: {file}")
        print(e)
        # Print stack trace
        print(traceback.format_exc())

def process_folder(folder):
    if debug_check(LVL_INFO):
        print(f"Processing Folder: {folder}")

    # Get extensions, adding prefix if necessary
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in get_globals("extension").split(",")]

    # Get all video files in lexical order
    all_videos = [file for file in folder.glob("*") if file.suffix in extensions]
    all_videos.sort()

    # Loop through all the files
    for file in all_videos:
        # Process the file
        process_file(file)

    # Recurse into subdirectories
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            process_folder(subfolder)

def validate_environment():
    # Validate topaz video ai is installed
    path = get_tvai_path()
    if not path.exists():
        raise Exception(f"Topaz Video Enhance AI missing: {str(path)}")

    # Validate ffmpeg binary is available
    if not get_ffmpeg().exists():
        raise Exception(f"ffmpeg missing: {str(path)}")

    # Validate ffprobe binary is aviailable
    if not get_ffprobe().exists():
        raise Exception(f"ffprobe missing: {str(path)}")

    # Validate topaz video ai data directory is available
    path = get_tvai_data_path()
    if not path.exists():
        raise Exception(f"Topaz Video Enhance AI Data Directory missing: {str(path)}")
    if not (path / "models").exists():
        raise Exception(f"Topaz Video Enhance AI Data Directory missing models: {str(path / 'models')}")

    # Validate mkvtoolnix is installed
    path = get_mkvextract_path()
    if not path.exists():
        raise Exception(f"MKVToolNix missing: {str(path)}")

    # Validate mkvextract binary is available
    if not get_mkvextract().exists():
        raise Exception(f"mkvextract missing: {str(path)}")

    # Validate mkvmerge binary is available
    if not get_mkvmerge().exists():
        raise Exception(f"mkvmerge missing: {str(path)}")

def get_profile_yaml():
    return "profiles.yml"

def get_tvai_path():
    return get_globals("tvai_path")

def get_tvai_data_path():
    return get_globals("tvai_data_path")

def get_mkvextract_path():
    return get_globals("mkvtoolnix_path")

def get_ffmpeg():
    return get_tvai_path() / "ffmpeg.exe"

def get_ffprobe():
    return get_tvai_path() / "ffprobe.exe"

def get_mkvextract():
    return get_mkvextract_path() / "mkvextract.exe"

def get_mkvmerge():
    return get_mkvextract_path() / "mkvmerge.exe"

def debug_check(level):
    return get_globals("debug") >= level

def get_globals(name=None):
    global g_globals
    if name is not None:
        return g_globals[name]
    return g_globals

def init_globals(globals):
    global g_globals

    g_globals = globals

def main():
    def arg_debug_level(value):
        # Example validator that expects a numeric debug level
        if not value.isdigit():
            raise argparse.ArgumentTypeError("Expected a numeric debug level")
        return int(value)

    parser = argparse.ArgumentParser(description="Upconvert Video")
    parser.add_argument("-f", "--force", action="store_true", help="Force reprocessing of files")
    parser.add_argument("-d", dest='debug', default=0, action='store_const', const=1, help="Enable debug output (debug level 1)")
    parser.add_argument("--debug", default=0, type=arg_debug_level, help="Enable debug output at a specific verbosity")
    parser.add_argument("--ext", "--extension", dest="extension", help="File extensions to process [multiple extensions seperated by ',']", default=".mkv")
    parser.add_argument("--open", action="store_true", help="Open the output file on completion")
    parser.add_argument("-o", "--output-dir", dest="output_dir", help="Output directory for processed files", default="output")

    parser_group = parser.add_argument_group("Output Settings")
    parser_group.add_argument("-n", "--rename", action="store_true", help="Rename output files to indicate the upscale resolution")
    parser_group.add_argument("-r", "--resolution", dest="resolution", help="Resolution to target upscale [480, 720, 960, 1080, 1440, 2160, default=720]",
                              default="720", choices=["1x", "2x", "4x", "720", "1080", "1440", "2160"])
    parser_group.add_argument("--pf", "--pixel_format", dest="pix_fmt", help="Set pixel format (sample aspect ratio correction) [source, square, default=square]", default="square", choices=["source", "square"])
    parser_group.add_argument("-s", "--subfix", action="store_true", help="Fix subtitle rip issues, prefer SRT over VobSub and no default English track")
    parser_group.add_argument("-p", "--padding", action="store_true", help="Pad out video file to standard aspect ratio")
    parser_group.add_argument("--upscaler", help="Select upscaler model [artemis, proteus, default=proteus]", default="proteus", choices=["artemis", "proteus"])
    parser_group.add_argument("--upscaler-profile", help="Select the upscaler profile from which to use the ffmpeg filter complex settings (see profiles.yml) [default=default]", default="default")
    parser_group.add_argument("--ff:qp", dest="ff_qp", help="FFMPEG QP value for output video [default=17]", default="17")

    parser_group = parser.add_argument_group("Path Overrides")
    parser_group.add_argument("-tvai", "--topaz-video-ai", dest="tvai_path", help="Path to Topaz Video Enhance AI installation")
    parser_group.add_argument("-tvaidata", "--topaz-video-ai-data", dest="tvai_data_path", help="Path to Topaz Video Enhance AI installation Data Directory")
    parser_group.add_argument("--mkvtoolnix", dest="mkvtoolnix_path", help="Path to mkvtoolnix installation where mkvextract is located")
    parser.add_argument("files", nargs="*", help="List of files and folders to process")
    args = parser.parse_args()

    # Get a list of files and folders from the arguments
    def fix_trailing_slash(path):
        if path[-1] == "\"":
            return path[:-1]
        return path
    files = [Path(fix_trailing_slash(f)) for f in args.files]

    # If no files specified use current working directory
    if len(files) == 0:
        files = [Path.cwd()]

    # Set the global paths
    tvai_path = Path(args.tvai_path) if args.tvai_path else Path("C:\Program Files\Topaz Labs LLC\Topaz Video AI")
    tvai_data_path = Path(args.tvai_data_path) if args.tvai_path else Path("C:\ProgramData\Topaz Labs LLC\Topaz Video AI")
    mkvtoolnix_path = Path(args.mkvtoolnix_path) if args.mkvtoolnix_path else Path("C:\Program Files\MKVToolNix")

    init_globals({
        "force": args.force,
        "debug": args.debug,
        "output_dir": args.output_dir,
        "extension": args.extension,
        "resolution": args.resolution,
        "pix_fmt": args.pix_fmt,
        "open": args.open,
        "subfix": args.subfix,
        "rename": args.rename,
        "padding": args.padding,
        "upscaler": args.upscaler,
        "upscaler_profile": args.upscaler_profile,
        "ff_qp": args.ff_qp,
        "tvai_path": tvai_path,
        "tvai_data_path": tvai_data_path,
        "mkvtoolnix_path": mkvtoolnix_path
    })

    validate_environment()

    for file in files:
        # If the file is a file then process the file
        if file.is_file():
            process_file(file)
        # If the file is a directory then process the folder
        elif file.is_dir():
            process_folder(file)
        else:
            print(f"Invalid file or folder: {file}")

if __name__ == "__main__":
    main()