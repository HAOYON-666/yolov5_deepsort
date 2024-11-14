import logging
from AIDetector_pytorch import Detector
import cv2
import argparse
from utils.general import find_filename


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    det = Detector()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logging.error("Error: Could not open video source.")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    logging.info('fps: %d', fps)
    frame_interval = int(1000 / fps) 

   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    
    ret, first_frame = cap.read()
    if not ret:
        logging.error("Error: Could not read from video source.")
        return
    output_width = int(first_frame.shape[1] * 0.5)  
    output_height = int(first_frame.shape[0] * (output_width / first_frame.shape[1]))  

    output_video=find_filename(args.save_video)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))

    index = 0
    if args.save_data:
        save_name="results.json"
    else:
        save_name=None 
    filename=find_filename(save_name)
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        
        result = det.feedCap(frame,filename)
        if 'frame' not in result:
            logging.error("Error: 'frame' key not found in result.")
            break
        result_frame = result['frame']

        
        resized_frame = cv2.resize(result_frame, (output_width, output_height))

        
        video_writer.write(resized_frame)
        logging.info("The %d frame has been written", index)
        index += 1
    logging.info(f"The video_result has been written to {output_video}")
    if args.save_data:
        logging.info(f"The target_information has been saved to {filename}")
    
    cap.release()
    video_writer.release()

def parse_opt():
    parser = argparse.ArgumentParser(description='Process a video with AIDetector and save the result.')
    # Add arguments
    parser.add_argument("--weights", type=str, default="weights/yolov5s.pt", help="Due to version limitations, the weight file for version 5.0 can be filled in here")
    parser.add_argument("--source", type=str, default="/data/workspace/zhaoyong/data/安全帽.mp4", help="The input video,you can choice file or 0(webcam)")
    parser.add_argument("--save_video", type=str, default="result.mp4", help="Path to save the output video file.")
    parser.add_argument("--save_data", action='store_true', help="If you need to save the detection data, you can add this parameter, which will generate a. json file")
    parser.add_argument("--category", nargs='*' , default=None, help="Set the categories to be checked out, if left blank, all yolo categories will be selected.")
    
    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_opt()
    main(args)
