from image_recognition import Camera

def main():
    cam:Camera = Camera(debug=True)
    index = 0
    while(1):
        cam.Test(not index % 100 == 0)
        index += 1
    cam.close()

main()