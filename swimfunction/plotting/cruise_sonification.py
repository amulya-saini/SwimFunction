''' Converts cruise angle position amplitudes into music.

If you want to use this script, be sure to include the bonus dependencies below
    FluidSynth
        https://www.fluidsynth.org/
    audiolazy
        pip install audiolazy
    midiutil
        pip install midiutil
    midi2audio
        pip install midi2audio

Tones emit from each angle position whenever the position completes a waveform.
If the amplitude at the position is greater than control fish mean at that position,
the pitch drops proportionally.

    Inspiration:
Ideas based somewhat on
https://medium.com/@astromattrusso/sonification-101-how-to-convert-data-into-music-with-python-71a6dd67751c
and this script was originally a modification of
https://github.com/SYSTEMSounds/sonification-tutorials/blob/main/data2midi-part1.py
by Matt Russo, which has no license to cite.
'''

from collections import namedtuple
from typing import List, Union
import subprocess
import pathlib
import numpy
# https://github.com/danilobellini/audiolazy/blob/master/audiolazy/lazy_midi.py
from audiolazy import str2midi # https://pypi.org/project/audiolazy/
# https://github.com/MarkCWirt/MIDIUtil
from midiutil import MIDIFile # https://midiutil.readthedocs.io/en/1.2.1/
from midi2audio import FluidSynth

from swimfunction.data_access import PoseAccess
from swimfunction.data_models.Fish import Fish
from swimfunction.recovery_metrics.metric_analyzers import series_to_waveform_stats
from swimfunction.recovery_metrics.metric_analyzers.CruiseWaveformCalculator\
    import CruiseWaveformCalculator
from swimfunction import FileLocations
from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.video_processing import extract_frames, fp_ffmpeg
from swimfunction.plotting.matplotlib_helpers import get_cmap_values
from swimfunction.video_processing.CropTracker import CropTrackerLog, find_existing_logfile
from swimfunction.global_config.config import config


WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

duration_beats = 52.8 #desired duration in beats (1 beat = 1 second if bpm=60)
bpm = 60 #tempo (beats per minute)

y_scale = 0.5  #scaling parameter for y-axis data (1 = linear)

class KeypointToneGroup:
    ''' Tones for each angle position on the fish,
    going from rostral to caudal.
    '''
    arbitrary_chord = [
        'C5', 'G4', 'E4', 'C4',
        'A4', 'F3', 'E3', 'C2'
    ]
    # Rostral positions play G, caudal positions play C
    # Recently injured fish play a tritone instead of power chord,
    # which is quite striking!
    power_chord = [
        'G4', 'G4', 'G4', 'G4',
        'G4', 'C4', 'C4', 'C4'
    ]
    harmonic_series = [
        'E6', # Replaced D6 to be prettier
        'C5',
        'G4',
        'E4',
        'C4',
        'G3',
        'C3',
        'C2'
    ]

# MIDI 11 is the music box tone.
# Would have preferred Wintergatan's music box, but this will do.
# https://en.wikipedia.org/wiki/General_MIDI for instrument programs
# Set different programs for each keypoint if you want an orchestra of sound.
CHANNEL_PROGRAMS = [11 for _ in range(8)]

# This was empirically derived from the Tracking Experiment.
MAX_ANGLES_PER_KEYPOINT = numpy.asarray([
    0.68994162, 0.7532861, 0.91151385, 0.91868098,
    0.90348546, 0.95366362, 1.34497713, 1.43281487])

VEL_MIN, VEL_MAX = 35, 127   #minimum and maximum note velocity

##############################################################################
def map_value(value, min_value, max_value, min_result, max_result):
    '''maps value (or array of values) from one range to another'''
    result = min_result + (value - min_value) / (max_value - min_value) * (max_result - min_result)
    return result
##############################################################################

FPS = 19 # frames per second
FPM = FPS * 60 # frames per minute

WaveformStatsIndexed = namedtuple(
    'WaveformStatsIndexed',
    ['argpeaks', 'periods', 'frequencies', 'amplitudes']
)
PeakMidiSummary = namedtuple(
    'PeakMidiSummary',
    ['channel', 'pitch', 'time', 'duration', 'volume', 'pitchwheel']
)

DOT_LENGTH = 10 # pixels, diameter of dot

def gkern(l, sig):
    ''' Author: clemisch
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    creates gaussian kernel with side length `l` and a sigma of `sig`
    '''
    ax = numpy.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = numpy.exp(-0.5 * numpy.square(ax) / numpy.square(sig))
    kernel = numpy.outer(gauss, gauss)
    return kernel / numpy.sum(kernel)

def get_dot(color):
    ''' Image of a dot
    '''
    patch = numpy.full((DOT_LENGTH, DOT_LENGTH, 3), 0, dtype=numpy.uint8)
    kern = (gkern(l=DOT_LENGTH, sig=1)*255).astype(numpy.uint8)
    patch[kern > 0, 0] = color[0]
    patch[kern > 0, 1] = color[1]
    patch[kern > 0, 2] = color[2]
    return patch

def get_waveform_stats_from_series_with_argpeaks(
        one_series: numpy.ndarray,
        min_waveforms_per_cruise: int) -> WaveformStatsIndexed:
    ''' Gets all measurable periods and amplitudes for each dimension in series.

    Parameters
    ----------
    one_series : numpy.ndarray
        NxD array where N is number of observations and D is number of measured dimensions
    min_waveforms_per_cruise : int
        If there is no dimension that has at least this number of waveforms, None is returned.

    Returns
    -------
    WaveformStatsIndexed : namedtuple, None
        (argpeaks, periods, frequencies, amplitudes) or None if does not have min_waveforms_per_cruise
    '''
    series = numpy.asarray(one_series)
    if len(series.shape) == 1:
        series = series[:, numpy.newaxis]
    periods = [[] for _ in range(series.shape[1])]
    amplitudes = [[] for _ in range(series.shape[1])]
    argpeaks = [[] for _ in range(series.shape[1])]
    for d in range(series.shape[1]):
        ap = series_to_waveform_stats.remove_false_waves(
            series_to_waveform_stats.get_alternating_argextrema(
                series[:, d]).astype(int), series[:, d])
        if len(ap) < 3:
            continue
        argpeaks[d] = ap[1:ap.size - 1:2]
        periods[d] = series_to_waveform_stats.argpeaks_to_periods(ap)
        peaks2d = numpy.asarray([ap, numpy.take(series[:, d], ap)]).T
        amplitudes[d] = series_to_waveform_stats.peaks_to_amplitudes2d(peaks2d)
        assert argpeaks[d].size == periods[d].size, f'({ap.size}) {argpeaks[d].size} != {periods[d].size}'
    if max([len(x) for x in periods]) < min_waveforms_per_cruise:
        return None
    frequencies = list(map(series_to_waveform_stats.periods_to_frequencies, periods))
    return WaveformStatsIndexed(argpeaks, periods, frequencies, amplitudes)

def trim_wafeform_stats_argpeaks_positive(wf_stats: WaveformStatsIndexed, nframes):
    ''' Keep only the peaks within the temporal limits [0, nframes)
    '''
    for dim in range(len(wf_stats.argpeaks)):
        locs = numpy.where((wf_stats.argpeaks[dim] >= 0) & (wf_stats.argpeaks[dim] < nframes))
        wf_stats.argpeaks[dim] = wf_stats.argpeaks[dim][locs]
        wf_stats.periods[dim] = wf_stats.periods[dim][locs]
        wf_stats.frequencies[dim] = wf_stats.frequencies[dim][locs]
        wf_stats.amplitudes[dim] = wf_stats.amplitudes[dim][locs]
    return wf_stats

class ControlComparer:
    ''' Compares keypoints to control averages
    '''
    def __init__(self):
        wf_calc = CruiseWaveformCalculator(
            config.access_params.experiment_name,
            AnnotationTypes.predicted)
        self.control_means = []
        comparisons = [[] for _ in range(8)]
        for assay in WPI_OPTIONS:
            df = wf_calc.get_waveform_stats(None, assay, None, None)
            for keypoint in range(8):
                assay_keypoint_mean = df.loc[:, (keypoint, 'amplitudes', 'mean')].mean()
                if assay == -1:
                    self.control_means.append(assay_keypoint_mean)
                comparisons[keypoint].append(self.compare_to_control(
                    keypoint,
                    df.loc[:, (keypoint, 'amplitudes', 'mean')]))
        all_comparisons = numpy.concatenate([numpy.concatenate(c) for c in comparisons]).flatten()
        ## Below: min/max comparisons any keypoint
        self.keypoint_comparison_min = all_comparisons.min()
        self.keypoint_comparison_max = all_comparisons.max()
        ## Below: min/max comparisons per keypoint
        # for keypoint in range(8):
        #     keypoint_comparisons = numpy.concatenate(comparisons[keypoint])
        #     self.keypoint_comparison_mins.append(keypoint_comparisons.min())
        #     self.keypoint_comparison_maxs.append(keypoint_comparisons.max())

    def compare_to_control(self, keypoint, observed_amplitudes: Union[float, numpy.ndarray]):
        ''' Calculates only the amount greater than control mean.
        '''
        return numpy.clip(
            observed_amplitudes - self.control_means[keypoint],
            a_min=0,
            a_max=numpy.inf)

    def compare_to_control_normalized(self, keypoint, observed_amplitudes: Union[float, numpy.ndarray]):
        ''' Normalize to the minimum and maximum comparisons in the experiment.
        '''
        return map_value(
            self.compare_to_control(keypoint, observed_amplitudes),
            self.keypoint_comparison_min, #s[keypoint],
            self.keypoint_comparison_max, #s[keypoint],
            0,
            1)

class CruiseMidiMaker:
    def __init__(self, fish_name, assay_label, start_frame, nframes, savedir):
        self.fish_name = fish_name
        self.assay_label = assay_label
        self.start_frame = start_frame
        self.nframes = nframes
        self.end_frame = start_frame + nframes - 1
        self.poses = []
        self.coordinate_poses = []
        self.wf_stats = []
        self.keypoint_frame_sounds = [[] for _ in range(8)]
        self.savedir = pathlib.Path(savedir)

    def save_video(self, keypoint_notes=KeypointToneGroup.harmonic_series):
        self._save_audio(keypoint_notes)
        self._save_accompanying_video()
        self._combine_audio_video()

    def _filepath(self, suffix='.mid', basename_tag=''):
        basename =  f'{basename_tag}{self.fish_name}_{self.assay_label}_{self.start_frame}_{self.nframes}_{FPS}fps.mid'
        return (self.savedir / basename).with_suffix(suffix)

    def _save_audio(self, keypoint_notes):
        self._calculate_waveform_stats()
        peak_midi_summaries = self._waveforms_to_time_midi_velocity(keypoint_notes)
        frames_per_quarter_note = 4
        num_keypoints = len(keypoint_notes)
        midi_writter = MIDIFile(
            numTracks=num_keypoints + 1,
            adjust_origin=False,
            ticks_per_quarternote=frames_per_quarter_note,
            eventtime_is_ticks=True)
        midi_writter.addTempo(track=0, time=0, tempo=FPM / frames_per_quarter_note)
        # Force the correct duration
        midi_writter.addNote(
            track=num_keypoints, channel=num_keypoints, time=0,
            pitch=str2midi('C2'), duration=self.nframes, volume=0)
        # Set instruments
        for channel, midi_program in enumerate(CHANNEL_PROGRAMS):
            midi_writter.addProgramChange(channel, channel, 0, midi_program)
        # Write notes
        for channel, pitch, time, duration, volume, pitchwheel in peak_midi_summaries:
            # Each keypoint has its own channel because pitch wheel affects all notes in a channel.
            midi_kwargs = dict(track=channel, channel=channel, time=time)
            midi_writter.addNote(pitch=pitch, duration=duration, volume=volume, **midi_kwargs)
            midi_writter.addPitchWheelEvent(pitchWheelValue=pitchwheel, **midi_kwargs)
        # Write the midi and wav files
        midifile = self._filepath('.mid')
        with open(midifile, 'wb') as fh:
            midi_writter.writeFile(fh)
        self._midi_to_wav()
        print('Saved', midifile)
        return self

    def _calculate_waveform_stats(self) -> WaveformStatsIndexed:
        ''' Gets poses, finds extrema. Returns amplitudes and timestamps.
        '''
        swim = Fish(self.fish_name).load()[self.assay_label]
        self.poses = PoseAccess.get_feature_from_assay(
            swim, feature='smoothed_angles', filters=[], keep_shape=True)
        self.coordinate_poses = PoseAccess.get_feature_from_assay(
            swim, feature='smoothed_coordinates', filters=[], keep_shape=True)
        fbuffer = 5
        self.wf_stats = get_waveform_stats_from_series_with_argpeaks(
            self.poses[self.start_frame-fbuffer:self.end_frame+fbuffer+1],
            min_waveforms_per_cruise=0)
        for i in range(len(self.wf_stats.argpeaks)):
            self.wf_stats.argpeaks[i] -= fbuffer
        self.wf_stats = trim_wafeform_stats_argpeaks_positive(self.wf_stats, self.nframes)

    def _waveforms_to_time_midi_velocity(self, keypoint_notes) -> List[PeakMidiSummary]:
        comparer = ControlComparer()
        peak_midi_summaries = []
        for keypoint, note in enumerate(keypoint_notes):
            for i in range(1, len(self.wf_stats.amplitudes[keypoint])):
                amplitude = numpy.nanmean(self.wf_stats.amplitudes[keypoint][i-1:i+1])
                time = int(self.wf_stats.argpeaks[keypoint][i])
                duration = int(self.wf_stats.argpeaks[keypoint][i] \
                    - self.wf_stats.argpeaks[keypoint][i-1])
                if amplitude is not None and not numpy.isnan(amplitude):
                    volume = int(map_value(
                        amplitude,
                        min_value=0,
                        max_value=MAX_ANGLES_PER_KEYPOINT[keypoint],
                        min_result=VEL_MIN,
                        max_result=VEL_MAX))
                    flatter_amt = comparer.compare_to_control_normalized(keypoint, amplitude)
                    pitchwheel = 0
                    if flatter_amt > 0:
                        pitchwheel = int(-map_value(flatter_amt, 0, 1, 0, 8192))
                    self.keypoint_frame_sounds[keypoint].append(time)
                    peak_midi_summaries.append(
                        PeakMidiSummary(
                            keypoint,
                            str2midi(note),
                            time,
                            duration,
                            volume,
                            pitchwheel))
        return peak_midi_summaries

    def _midi_to_wav(self):
        midifile = self._filepath('.mid')
        wavfile = self._filepath('.wav')
        soundfont = pathlib.Path('/Users/nick/Box/dissertation/sonification/Creative(emu10k1)8MBGMSFX.SF2')
        fs = FluidSynth()
        if soundfont.exists():
            fs = FluidSynth(sound_font=soundfont.as_posix())
        fs.midi_to_audio(midifile.as_posix(), wavfile.as_posix())
        print('Saved', wavfile)

    def _save_accompanying_video(self):
        ct_log = CropTrackerLog().read_from_file(find_existing_logfile(FileLocations.find_video(self.fish_name, self.assay_label)))
        frames = extract_frames.extract_frames(
            self.fish_name,
            self.assay_label,
            range(self.start_frame, self.end_frame+1),
            frame_nums_are_full_video=True,
            force_grayscale=True)
        imgs = []
        for i, frame in enumerate(frames):
            imgs.append(self._annotate_frame(i, frame, ct_log))
        fp_ffmpeg.write_video(
            imgs,
            FPS,
            outpath=self._filepath('.avi').as_posix())
        return self

    def _annotate_frame(self, frame_i, frame, ct_log: CropTrackerLog):
        frame_num = self.start_frame + frame_i
        corner = ct_log.get_corner(frame_num)
        def to_uint8(v):
            return int(map_value(v, 0, 1, 0, 255))
        img = frame
        if len(frame.shape) == 2:
            img = numpy.dstack([frame, frame, frame]).reshape((*frame.shape, 3)).astype(numpy.uint8)
        keypoint_colors = [ numpy.asarray([to_uint8(v) for v in c[:3]]) for c in reversed(get_cmap_values('rainbow', 8))]
        for keypoint in range(8):
            if frame_i + 1 in self.keypoint_frame_sounds[keypoint] \
                    or frame_i in self.keypoint_frame_sounds[keypoint] \
                    or frame_i - 1 in self.keypoint_frame_sounds[keypoint]:
                kp = self.coordinate_poses[frame_num][keypoint+1] - corner
                kp_tr = (int(kp[0]-(DOT_LENGTH/2)), int(kp[1]-(DOT_LENGTH/2)))
                dot = get_dot(keypoint_colors[keypoint])
                patch = img[
                        kp_tr[0]:kp_tr[0]+DOT_LENGTH,
                        kp_tr[1]:kp_tr[1]+DOT_LENGTH,
                    :].astype(float)
                patch[dot > 0] = dot[dot > 0]
                img[
                    kp_tr[0]:kp_tr[0]+DOT_LENGTH,
                    kp_tr[1]:kp_tr[1]+DOT_LENGTH,
                    :] = patch.astype(numpy.uint8)
        return img

    def _combine_audio_video(self):
        # do not allow std input,
        # otherwise it would annoyingly suspend the process.
        ff_cmd = [
            fp_ffmpeg.FFMPEG_BIN,
            '-nostdin'
        ]
        ff_cmd = ff_cmd + [
            '-i', self._filepath('.wav').as_posix(),
            '-i', self._filepath('.avi').as_posix(),
            '-af', 'apad',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            '-y', # Replace existing
            self._filepath('.avi', 'combined_').as_posix()
        ]
        print(' '.join(ff_cmd))
        subprocess.call(ff_cmd, stdout=subprocess.PIPE)

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument('fish_name'),
        lambda parser: parser.add_argument('assay_label', type=int),
        lambda parser: parser.add_argument('start_frame', type=int),
        lambda parser: parser.add_argument(
            'nframes', type=int, help='Number of consecutive frames to process.'),
        lambda parser: parser.add_argument('savedir')
    )
    CruiseMidiMaker(
        ARGS.fish_name,
        ARGS.assay_label,
        ARGS.start_frame,
        ARGS.nframes,
        pathlib.Path(ARGS.savedir)).save_video()
