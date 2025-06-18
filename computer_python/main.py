from image_recognition import Camera
from path_find import track

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

main()