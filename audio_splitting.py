from pydub import AudioSegment
import feature_extraction
import io
def split_audio(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)

    segment_duration = 3 * 1000  # 3 seconds in milliseconds
    audio_duration = len(audio)

    # Check if the audio is shorter than 1 minute and 3 seconds
    if audio_duration < 63 * 1000:
        # If it's shorter, take audio from 0 to 3 seconds
        segment = audio[:segment_duration]
    else:
        # If it's longer, take audio from 1 minute to 1 minute 3 seconds
        start_time = 60 * 1000
        end_time = start_time + segment_duration
        segment = audio[start_time:end_time]

    output_stream = io.BytesIO()
    segment.export(output_stream, format="wav")
    output_stream.seek(0)

    # Process and extract features from the segment
    features = feature_extraction.all_feature_extraction(output_stream)
    return features