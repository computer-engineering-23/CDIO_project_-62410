from image_recognition import Camera

def main():
    cam:Camera = Camera()
    while(1):
        cam.Test()
    cam.close()

main()