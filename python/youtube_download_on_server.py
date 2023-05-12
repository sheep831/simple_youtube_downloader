import sys
import pytube

# Get the YouTube video URL from the command-line argument
if len(sys.argv) < 2:
    print("Usage: python youtube_downloader.py <video_url>")
    sys.exit(1)

video_url = sys.argv[1]

# Create a YouTube object and get the highest quality stream
yt = pytube.YouTube(video_url)
stream = yt.streams.get_highest_resolution()

# Download the video to the current directory
print("Downloading video...")
stream.download(output_path="/Users/Apex/Desktop/test_python_nodejs/videos")
print("Video downloaded successfully!")
