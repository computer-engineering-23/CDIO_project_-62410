from image_recognition import Camera
from path_find import track

def main():
    cam:Camera = Camera(debug=False)
    track_instance = track(cam)
    track_instance.update(walls=True, goals=True, targets=True, obsticles=True, car=True)
    track_instance.test()

main()