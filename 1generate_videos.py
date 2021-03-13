# vim: expandtab:ts=4:sw=4
import os
import argparse
import show_results


def convert(filename_in, filename_out, ffmpeg_executable="ffmpeg"):
    import subprocess
    command = [ffmpeg_executable, "-i", filename_in, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_out]
    subprocess.call(command)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)  #--sequence_dir보다 전dir까지 (train, test)
    parser.add_argument(
        "--result_dir", help="Path to the folder with tracking output.",
        required=True)  #--result/text
    parser.add_argument(
        "--output_dir", help="Folder to store the videos in. Will be created "
        "if it does not exist.",
        required=True)  #영상저장할 곳 새로지정 
    parser.add_argument(
        "--convert_h264", help="If true, convert videos to libx264 (requires "
        "FFMPEG", default=False)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    IDnum = int(input("ID_num for tracking (no foot tacking is 0) : "))
    

    os.makedirs(args.output_dir, exist_ok=True)
    print(os.listdir(args.result_dir))
    for sequence_txt in os.listdir(args.result_dir):    #sequence_txt는 MOT16-01.txt를 먼저
        sequence = os.path.splitext(sequence_txt)[0]    #MOT16-01
        sequence_dir = os.path.join(args.mot_dir, sequence)
        #with open(sequence_txt, 'r') as track:
        #    track_id = [int(l.split(',')[1]) for l in f_hi.read().splitlines()[:]]
        
        if not os.path.exists(sequence_dir):
            continue
        result_file = os.path.join(args.result_dir, sequence_txt)
        update_ms = args.update_ms

        video_filename = os.path.join(args.output_dir, "%s_ID%s_tracking.avi" % (sequence, IDnum))
#        video_filename = os.path.join(args.output_dir, "%s_ID%s_foot_10.avi" % (sequence, IDnum))

        print("Saving %s to %s." % (sequence_txt, video_filename))
        
        show_results.run(
            sequence_dir, result_file, False, None, update_ms, video_filename, IDnum)



    if not args.convert_h264:
        import sys
        sys.exit()

    for sequence_txt in os.listdir(args.result_dir):
        sequence = os.path.splitext(sequence_txt)[0]
        sequence_dir = os.path.join(args.mot_dir, sequence)
        if not os.path.exists(sequence_dir):
            continue
        filename_in = os.path.join(args.output_dir, "%s.avi" % sequence)
        filename_out = os.path.join(args.output_dir, "%s.mp4" % sequence)
        convert(filename_in, filename_out)
