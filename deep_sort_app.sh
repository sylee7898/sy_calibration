python deep_sort_app.py \
    --sequence_dir=./data/MOT16/test/MOT16-01 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-01.npy \
    --output_file=./result/text/MOT16-01.txt \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True

python generate_videos.py \
    --mot_dir=./data/MOT16/test \
    --result_dir=./result/text \
    --output_dir=./result/video
