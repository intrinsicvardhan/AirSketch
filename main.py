import cv2

#start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    #Display the captured frame
    cv2.imshow('Live Drawing', frame)

    #Break the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release() 
cv2.destroyAllWindows()

# from detect_color import detect_color

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break


#     detected_color_img, mask = detect_color(frame)

#     cv2.imshow('Detected Color', detected_color_img)
#     cv2.imshow('Mask', mask)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()

# from draw_contours import find_and_draw_contours

# while True: 
#     ret, frame = cap.read()
#     if not ret: 
#         break

#     _, mask = detect_color(frame)

#     drawing = find_and_draw_contours(frame, mask)

#     cv2.imshow('Drawing', drawing)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# from predict_chars import predict_character



# #start main code
# def main():
#     while True:
#         ret, frame = cap.read()
#         if not ret: 
#             break

#         _, mask = detect_color(frame)
#         drawing = find_and_draw_contours(frame, mask)

#         character = predict_character(mask)

#         cv2.putText(drawing, 'Detected: ' + str(character), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow('Drawing', drawing)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


# main()
