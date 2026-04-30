import cv2


def test_source(index, backend_name, backend_flag):
    cap = cv2.VideoCapture(index, backend_flag)
    opened = cap.isOpened()
    frame_ok = False

    if opened:
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_ok = True
                break

    cap.release()
    return opened, frame_ok, backend_name


def main():
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]

    found = []
    print("Scanning camera indexes 0-5...")

    for index in range(6):
        for backend_name, backend_flag in backends:
            opened, frame_ok, backend = test_source(index, backend_name, backend_flag)
            print(
                f"Camera {index} with {backend}: "
                f"opened={opened}, frame_ok={frame_ok}"
            )
            if opened and frame_ok:
                found.append((index, backend))

    if found:
        print("\nWorking camera sources:")
        for index, backend in found:
            print(f"CAMERAS = [{index}]  with backend {backend}")
    else:
        print("\nNo working local webcam source was found.")
        print("If you are using CCTV/IP cameras, use RTSP/HTTP stream URLs instead of indexes.")


if __name__ == "__main__":
    main()
