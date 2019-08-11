while True:
      ret, image_np = cap.read() #if streaming image is small image as trained image
       cv2.imshow('object detection', cv2.resize(image_np, (800,600))) #resize the image
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      
