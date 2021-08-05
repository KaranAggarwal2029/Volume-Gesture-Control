import mediapipe as mp
import cv2
import time


class handDetector():
    def __init__(self,mode=False,maxHands =2, detectionCon =0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)  #default params - Static_image_node = False, max_num_hands=2 Means by this we tell how many hands you want to detect
        self.mpDraw = mp.solutions.drawing_utils  # To draw the line between all the 21 points
        
    
    def findHands(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  #Used to process the frame and give the result to us
         
        
        #To extract the info in the results
         
        #To extract multiple hands 1 by 1 using loop # First need to check whether there is something in the results
        #Checking whether there is hand or not
        
        #print(results.multi_hand_landmarks)  # Whenever u will bring hand it will show result else None

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                #mpDraw.draw_landmarks(frame,handlms)  Help to draw the 21 points on the hand
                
                 # We are going to get info within this hand
                #we will get the id no. and the landmark information(will give x and y coordinates)
                 
                #Now to connect them
                if draw:
                    self.mpDraw.draw_landmarks(frame,handlms,self.mpHands.HAND_CONNECTIONS)
                
        return frame
    
    
    def findPosition(self, frame, handNo=0,draw = True):
        
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhands.landmark):
                
                #print(id, lm)
                #This will give us x,y,z coordinates(ratio of the image so we need ot convert it by multiplying with width and height) but we need only x,y to locate the particular feature 
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y * h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                
                if draw:
                    cv2.circle(frame,(cx,cy),5,(0,0,255), cv2.FILLED)
        return lmList
                
        


def main():
    
    
    pTime = 0 #Previous Time
    cTime = 0 # Current Time
    cap = cv2.VideoCapture(0) 
    detector = handDetector()
    
    while True:
        red, frame = cap.read()
        frame = detector.findHands(frame)
        detector.findPosition(frame)
        
        #For the frame rate
        cTime = time.time()
        fps = float(1/(cTime - pTime))
        pTime = cTime
        cv2.putText(frame,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
            
        
        cv2.imshow("Image", frame)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    


    

if __name__ == "__main__":
    main()