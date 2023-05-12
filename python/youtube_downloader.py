import sys
import os
import pytube

if len(sys.argv) < 2:
    print('Please provide a valid YouTube video URL as an argument.')
    sys.exit()

url = sys.argv[1]
try:
    video = pytube.YouTube(url)
except:
    print('Invalid YouTube URL provided. Please try again.')
    sys.exit()

stream = video.streams.get_highest_resolution()
print(f'Downloading {video.title}...')
stream.download(output_path=os.path.expanduser("~") + "/Downloads")
print(f'{video.title} downloaded successfully to Downloads folder.')


