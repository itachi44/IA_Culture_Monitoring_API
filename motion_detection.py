import argparse
import datetime
import imutils
import time
import numpy as np
import cv2
import sys


# Globals
log_file = None # Log file

# Custom version of print
def lprint(*objects,sep=" ", end='\n', file=sys.stdout, flush=False): 
    print(*objects, sep=sep, end=end, file=file, flush=flush)
    if log_file is not None and not log_file.closed:
        print(*objects, sep=sep, end=end, file=g_log_file, flush=flush)


def aquire_args():
    
    global args

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(description='Program motion capture program')
    parser.add_argument("-v", "--video", help="path to the video file")
    parser.add_argument("-a", "--minArea", type=int, default=500, help="minimum area size")
    parser.add_argument('-l','--log-file', help='Sets path for log file.')
    args = parser.parse_args()


# Check if arguments are valid
    
def check_args():
    
    # Check arg: watermark or watermark text specified
    if args.video is not None and len(args.video) == 0:
        raise Exception(1,"Vidéo non spécifiée.")
    elif args.minArea is not None and args.minArea == 0:
        raise Exception(10,"taille de région incorrecte.")



def motion_detector():

    # if the video argument is None, then we are reading from webcam
    if args.video is None:
        vs = cv2.VideoCapture(0)
            
    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(args.video)

    # Check if the webcam is opened correctly
    if not vs.isOpened():
        raise IOError("Cannot open webcam")
    
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    size = (frame_width, frame_height)

    previous_frame = None
    count = 0 #frequence de capture
    result = cv2.VideoWriter('./metadata/saved_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'),10, size)

    while vs.isOpened():

        # 1. Load image; convert to RGB
        ret, frame = vs.read()
        if ret == True:
            
            img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

            result.write(frame)
            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)
            # 2. Calculate the difference
            if (previous_frame is None):
            # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # 3. Set previous frame and continue if there is None
            if (previous_frame is None):
            # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>25 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]

            # 6. Find and optionally draw contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            record = False
            for contour in contours:
                if cv2.contourArea(contour) < 1000 :
                # too small: skip!
                    record = False
                    continue 
                (x, y, w, h) = cv2.boundingRect(contour)
                #cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
                record = True

            if record == True:
                #vs.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # wait 1 sec between each capture
                cv2.imwrite('./metadata/detected_images/detected_img.jpg',img_rgb[:, :, ::-1])      # save frame as JPG file

            cv2.imshow('Motion detector', frame)


            #press esc to quit
            if (cv2.waitKey(30) == 27):
                break

        else :
            break
    # Cleanup
    vs.release()
    cv2.destroyAllWindows()

          

def main():

    global log_file

    try:
        aquire_args()
        check_args()
        lprint("running...")

        motion_detector()

        # Open log file if requested
        if args.log_file is not None and len(args.log_file) > 0:
            log_file = open(args.log_file,"a")

    except (Exception,SystemExit,IOError) as e:
        if len(e.args) >= 2:
            lprint(e.args[1] + " (Error code: %i)" % (e.args[0]))
        if len(e.args) >= 1:
            sys.exit(e.args[0])
    except:
        lprint('Unexpected error: ',sys.exc_info()[0])
        raise
        
    try:
        if log_file is not None and not log_file.closed:
            log_file.close()
    except (IOError):
        pass



if __name__ == "__main__":
    main()
else:
    lprint("Script not run properly")
    sys.exit(1)