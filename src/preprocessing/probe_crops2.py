import io, os, sys, zipfile
import numpy as np
import cv2
import av

DATA = r"C:\dev\data\avcaffe"
clip = sys.argv[1]
p, rest = clip.split("_task_")
t = rest.split("_clip_")[0]
zp = os.path.join(DATA, "face_crops", p + "_task_" + t, clip + ".zip")
vp = os.path.join(DATA, "videos", "shorter_segments", p, "task_" + t, clip + ".avi")


def describe(data, depth=0):
    pad = "  " * depth
    print(pad + "layer", depth, "| bytes:", len(data), "| first bytes:", data[:8])
    if data[:6] == b"\x93NUMPY":
        a = np.load(io.BytesIO(data))
        print(pad + "npy:", a.shape, a.dtype)
        return
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        exts = {}
        for n in names:
            e = os.path.splitext(n)[1].lower()
            exts[e] = exts.get(e, 0) + 1
        print(pad + "entries:", len(names), "| extensions:", exts)
        print(pad + "first 3:", names[:3])
        print(pad + "last 2:", names[-2:])
        print(pad + "uncompressed MB: %.2f" % (sum(i.file_size for i in infos) / 1e6))
        imgs = [n for n in names if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if imgs:
            shapes = set()
            for n in imgs[:: max(1, len(imgs) // 10)]:
                im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                shapes.add(None if im is None else im.shape)
            print(pad + "frame shapes (sampled):", shapes)
        elif names and depth < 4:
            describe(z.read(names[0]), depth + 1)


with open(zp, "rb") as f:
    describe(f.read())
with av.open(vp) as c:
    v = c.streams.video[0]
    print("clip video frames:", v.frames, "| fps:", v.average_rate)
