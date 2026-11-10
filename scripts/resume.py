"""Lanjutkan training dari last.pt (resume). Pemakaian: resume.py NAMA_RUN WORKERS"""
import sys, time, json, pathlib
from ultralytics import YOLO


def main():
    name, workers = sys.argv[1], int(sys.argv[2])
    project = pathlib.Path("football_detection/results/train").resolve()
    t = time.time()
    YOLO(str(project / name / "weights/last.pt")).train(resume=True, workers=workers)
    json.dump(dict(resumed=True, workers_after_resume=workers, resume_segment_seconds=round(time.time() - t, 1)),
              open(project / name / "resume_info.json", "w"), indent=1)
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
