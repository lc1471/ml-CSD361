import os, sys, zipfile, io
import numpy as np
import cv2
import av

DATA = r"C:\dev\data\avcaffe"
clip = sys.argv[1]
p, rest = clip.split("_task_")
t = rest.split("_clip_")[0]
zp = os.path.join(DATA, "face_crops", p + "_task_" + t, clip + ".zip")
vp = os.path.join(DATA, "videos", "shorter_segments", p, "task_" + t, clip + ".avi")

with zipfile.ZipFile(zp) as z:
    infos = [i for i in z.infolist() if not i.is_dir()]
    names = [i.filename for i in infos]
    print("zip entries:", len(names))
    print("first 3:", names[:3])
    print("last 2:", names[-2:])
    exts = {}
    for n in names:
        e = os.path.splitext(n)[1].lower()
        exts[e] = exts.get(e, 0) + 1
    print("extensions:", exts)
    print("uncompressed MB: %.2f" % (sum(i.file_size for i in infos) / 1e6))
    imgs = [n for n in names if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if imgs:
        shapes = set()
        for n in imgs[:: max(1, len(imgs) // 10)]:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            shapes.add(None if im is None else im.shape)
        print("frame shapes (sampled):", shapes)
    elif names and names[0].lower().endswith(".npy"):
        a = np.load(io.BytesIO(z.read(names[0])))
        print("npy shape/dtype:", a.shape, a.dtype)

with av.open(vp) as c:
    v = c.streams.video[0]
    print("clip video frames:", v.frames, "| fps:", v.average_rate)
