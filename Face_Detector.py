import cv2

# Load  pre-trained data on face frontals from opencv 
trainded_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
webcam = cv2.VideoCapture(0) 


# Iterate over the frames forever
while True:

   ### Read the current frame
    successful_frame_read, frame = webcam.read() 

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_coordinates = trainded_face_data.detectMultiScale(grayscaled_img)


    # draw rectangles around the faces
    for (x, y, w, h) in faces_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 8)
    # Load an image to detect faces in
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Force quit with Q
    if key==81 or key==113:
        break

webcam.release()

print("Completed")






