import os, sys
import av

DATA = r"C:\dev\data\avcaffe"
path = os.path.join(DATA, sys.argv[1])
with av.open(path) as c:
    v = c.streams.video[0]
    print("video:", v.codec_context.name, v.average_rate, "fps,", v.width, "x", v.height, "| frames:", v.frames)
    print("audio streams:", len(c.streams.audio))
    for a in c.streams.audio:
        print("audio:", a.codec_context.name, a.rate, "Hz,", a.channels, "ch")
    print("duration (s):", c.duration / 1e6 if c.duration else None)
