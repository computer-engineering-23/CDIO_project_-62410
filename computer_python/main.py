from image_recognition import Camera
from path_find import track
import Log

# def main():
#     Log.enableLog()
#     Log.printLog("INFO", "Logging enabled",producer="main")
#     Log.blockTag("DEBUG")
#     Log.printLog("debug", "not visible", producer="missing")
#     Log.printLog("INFO", "Starting main function", producer="test main")
#     Log.blockTag("info")
#     Log.closeLog()
def main():
    cam:Camera = Camera(debug=True)
    index = 0
    while(1):
        cam.Test(not index % 100 == 0)
        index += 1
    cam.close()
# def main():
#     cam:Camera = Camera(debug=True)
#     track_instance = track(cam)
#     track_instance.update(walls=True, goals=True, targets=True, obsticles=True, car=True)
#     track_instance.test()


# def run_slider_mode():
#     cam = Camera(debug=True)
#     cam.adjustWithSliders()

# def run_tracking_mode():
#     cam = Camera(debug=False)
#     track_instance = track(cam)
#     track_instance.update(walls=True, goals=True, targets=True, obsticles=True, car=True)
#     track_instance.test()

# def main(mode="slider"):
#     if mode == "slider":
#         run_slider_mode()
#     else:
#         run_tracking_mode()

# if __name__ == "__main__":
#     main(mode="slider")  # Change to 'tracking' to run full system
main()